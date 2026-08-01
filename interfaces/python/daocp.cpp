#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include "src.c"
}

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

Array fixed_array(py::handle value, const char* name,
                  std::initializer_list<py::ssize_t> shape) {
    Array out = Array::ensure(value);
    if (!out)
        throw py::type_error(std::string(name) + " must be a real-valued array");
    if (out.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw py::value_error(std::string(name) + " has the wrong number of dimensions");
    py::ssize_t axis = 0;
    for (py::ssize_t size : shape) {
        if (out.shape(axis) != size)
            throw py::value_error(std::string(name) + " has an inconsistent shape");
        ++axis;
    }
    return out;
}

struct Ragged {
    Array dense;
    std::vector<double> values;
    std::vector<u32> rows;
    bool dense_backed = false;

    f64* data() {
        return dense_backed ? const_cast<f64*>(dense.data()) : values.data();
    }
};

Ragged ragged_matrix(py::handle value, u32 N, u32 columns, const char* name) {
    Ragged out;
    out.rows.reserve(N);

    if (py::isinstance<py::array>(value)) {
        out.dense = Array::ensure(value);
        out.dense_backed = true;
        if (!out.dense)
            throw py::type_error(std::string(name) + " must be real-valued");
        if (out.dense.ndim() != 3 || out.dense.shape(0) != N ||
            out.dense.shape(2) != columns)
            throw py::value_error(std::string(name) + " must have shape (N, m[t], n)");
        if (out.dense.shape(1) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " has too many rows");
        out.rows.assign(N, static_cast<u32>(out.dense.shape(1)));
        return out;
    }

    if (!py::isinstance<py::sequence>(value))
        throw py::type_error(std::string(name) + " must be an array or a sequence of arrays");
    py::sequence stages = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(stages) != N)
        throw py::value_error(std::string(name) + " must contain N stages");
    for (u32 t = 0; t < N; ++t) {
        Array stage = Array::ensure(stages[t]);
        if (!stage || stage.ndim() != 2 || stage.shape(1) != columns)
            throw py::value_error(std::string(name) + " stage " + std::to_string(t) +
                                  " has an inconsistent shape");
        if (stage.shape(0) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " has too many rows");
        out.rows.push_back(static_cast<u32>(stage.shape(0)));
        out.values.insert(out.values.end(), stage.data(), stage.data() + stage.size());
    }
    return out;
}

Ragged ragged_vector(py::handle value, u32 N, const char* name) {
    Ragged out;
    out.rows.reserve(N);

    if (py::isinstance<py::array>(value)) {
        out.dense = Array::ensure(value);
        out.dense_backed = true;
        if (!out.dense)
            throw py::type_error(std::string(name) + " must be real-valued");
        if (out.dense.ndim() != 2 || out.dense.shape(0) != N)
            throw py::value_error(std::string(name) + " must have shape (N, m[t])");
        if (out.dense.shape(1) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " is too long");
        out.rows.assign(N, static_cast<u32>(out.dense.shape(1)));
        return out;
    }

    if (!py::isinstance<py::sequence>(value))
        throw py::type_error(std::string(name) + " must be an array or a sequence of arrays");
    py::sequence stages = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(stages) != N)
        throw py::value_error(std::string(name) + " must contain N stages");
    for (u32 t = 0; t < N; ++t) {
        Array stage = Array::ensure(stages[t]);
        if (!stage || stage.ndim() != 1)
            throw py::value_error(std::string(name) + " stage " + std::to_string(t) +
                                  " has an inconsistent shape");
        if (stage.shape(0) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " is too long");
        out.rows.push_back(static_cast<u32>(stage.shape(0)));
        out.values.insert(out.values.end(), stage.data(), stage.data() + stage.size());
    }
    return out;
}

void same_rows(const Ragged& matrix, const Ragged& bounds,
               const char* matrix_name, const char* bounds_name) {
    if (matrix.rows != bounds.rows)
        throw py::value_error(std::string(matrix_name) + " and " + bounds_name +
                              " must have the same number of rows at every stage");
}

bool symmetric(const double* A, u32 n) {
    double scale = 1.0;
    for (u32 i = 0; i < n * n; ++i) {
        if (!std::isfinite(A[i])) return false;
        scale = std::max(scale, std::abs(A[i]));
    }
    const double tol = 1e-10 * scale;
    for (u32 i = 0; i < n; ++i)
        for (u32 j = 0; j < i; ++j)
            if (std::abs(A[i * n + j] - A[j * n + i]) > tol) return false;
    return true;
}

bool cholesky_check(const double* A, u32 n, double regularization) {
    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    for (u32 i = 0; i < n; ++i) {
        for (u32 j = 0; j <= i; ++j) {
            double value = A[i * n + j];
            if (i == j) value += regularization;
            for (u32 k = 0; k < j; ++k) value -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (!std::isfinite(value) || value <= 0.0) return false;
                L[i * n + j] = std::sqrt(value);
            } else {
                L[i * n + j] = value / L[j * n + j];
            }
        }
    }
    return true;
}

double psd_epsilon(const double* A, u32 n) {
    double scale = 1.0;
    for (u32 i = 0; i < n; ++i) scale = std::max(scale, std::abs(A[i * n + i]));
    return 1e-10 * scale;
}

void check_cost(const Array& Q, const Array& R, const Array& S,
                u32 N, u32 nx, u32 nu) {
    const u32 nb = nx + nu;
    std::vector<double> block(static_cast<size_t>(nb) * nb);
    for (u32 t = 0; t < N; ++t) {
        const double* Qt = Q.data() + static_cast<size_t>(t) * nx * nx;
        const double* Rt = R.data() + static_cast<size_t>(t) * nu * nu;
        const double* St = S.data() + static_cast<size_t>(t) * nu * nx;
        if (!symmetric(Rt, nu) || !cholesky_check(Rt, nu, 0.0))
            throw py::value_error("R must be symmetric positive definite at every stage");
        if (!symmetric(Qt, nx) || !cholesky_check(Qt, nx, psd_epsilon(Qt, nx)))
            throw py::value_error("Q must be symmetric positive semidefinite at every stage");

        std::fill(block.begin(), block.end(), 0.0);
        for (u32 i = 0; i < nu; ++i) {
            for (u32 j = 0; j < nu; ++j) block[i * nb + j] = Rt[i * nu + j];
            for (u32 j = 0; j < nx; ++j) {
                block[i * nb + nu + j] = St[i * nx + j];
                block[(nu + j) * nb + i] = St[i * nx + j];
            }
        }
        for (u32 i = 0; i < nx; ++i)
            for (u32 j = 0; j < nx; ++j)
                block[(nu + i) * nb + nu + j] = Qt[i * nx + j];
        if (!cholesky_check(block.data(), nb, psd_epsilon(block.data(), nb)))
            throw py::value_error("[R S; S' Q] must be positive semidefinite at every stage");
    }
}

struct SolveInfo {
    u32 status;
    u32 iters;
    double solve_time;
};

struct SolveResult {
    py::array x;
    py::array u;
    SolveInfo info;
};

class OCPsolver {
public:
    OCPsolver(py::object A, py::object B, py::object w,
              py::object Q, py::object R, py::object S,
              py::object q, py::object r, py::object D,
              py::object C, py::object d, py::object c,
              py::object x0, u32 N, u32 nx, u32 nu, u32 max_iter)
        : initialized_(false), N_(N), nx_(nx), nu_(nu) {
        if (!N || !nx || !nu || !max_iter)
            throw py::value_error("N, nx, nu, and max_iter must be positive");

        Array Ap = fixed_array(A, "A", {N, nx, nx});
        Array Bp = fixed_array(B, "B", {N, nx, nu});
        Array wp = fixed_array(w, "w", {N, nx});
        Array Qp = fixed_array(Q, "Q", {N, nx, nx});
        Array Rp = fixed_array(R, "R", {N, nu, nu});
        Array Sp = fixed_array(S, "S", {N, nu, nx});
        Array qp = fixed_array(q, "q", {N, nx});
        Array rp = fixed_array(r, "r", {N, nu});
        Array x0p = fixed_array(x0, "x0", {nx});
        Ragged Dp = ragged_matrix(D, N, nx, "D");
        Ragged Cp = ragged_matrix(C, N, nu, "C");
        Ragged dp = ragged_vector(d, N, "d");
        Ragged cp = ragged_vector(c, N, "c");
        same_rows(Dp, dp, "D", "d");
        same_rows(Cp, cp, "C", "c");
        check_cost(Qp, Rp, Sp, N, nx, nu);

        workspace_init(&wrk_, const_cast<f64*>(Ap.data()), const_cast<f64*>(Bp.data()),
                       const_cast<f64*>(wp.data()), const_cast<f64*>(Qp.data()),
                       const_cast<f64*>(Rp.data()), const_cast<f64*>(Sp.data()),
                       const_cast<f64*>(qp.data()), const_cast<f64*>(rp.data()),
                       Dp.data(), Cp.data(), dp.data(), cp.data(),
                       const_cast<f64*>(x0p.data()), N, nx, nu,
                       Dp.rows.data(), Cp.rows.data(), max_iter);
        initialized_ = true;
    }

    ~OCPsolver() {
        if (initialized_) workspace_free(&wrk_);
    }

    OCPsolver(const OCPsolver&) = delete;
    OCPsolver& operator=(const OCPsolver&) = delete;

    SolveResult solve() {
        const auto start = std::chrono::steady_clock::now();
        const u32 iters = ::solve(&wrk_);
        const auto stop = std::chrono::steady_clock::now();
        const double micros = std::chrono::duration<double, std::micro>(stop - start).count();
        py::object owner = py::cast(this, py::return_value_policy::reference);
        SolveResult result{
            py::array_t<double>({N_ + 1, nx_}, {sizeof(double) * nx_, sizeof(double)},
                                wrk_.x, owner),
            py::array_t<double>({N_, nu_}, {sizeof(double) * nu_, sizeof(double)},
                                wrk_.u, owner),
            {wrk_.return_status, iters, micros}
        };
        return result;
    }

    void update(py::object x0, py::object q, py::object r,
                py::object A, py::object B, py::object w,
                py::object Q, py::object R, py::object S,
                py::object D, py::object C, py::object d, py::object c) {
        Array x0p = fixed_array(x0, "x0", {nx_});
        Array qp, rp, Ap, Bp, wp, Qp, Rp, Sp;
        Ragged Dp, Cp, dp, cp;

        const bool has_q = !q.is_none();
        const bool has_r = !r.is_none();
        if (has_q) qp = fixed_array(q, "q", {N_, nx_});
        if (has_r) rp = fixed_array(r, "r", {N_, nu_});

        const bool dynamics = !A.is_none() || !B.is_none() || !w.is_none();
        if (dynamics && (A.is_none() || B.is_none() || w.is_none()))
            throw py::value_error("A, B, and w must be updated together");
        if (dynamics) {
            Ap = fixed_array(A, "A", {N_, nx_, nx_});
            Bp = fixed_array(B, "B", {N_, nx_, nu_});
            wp = fixed_array(w, "w", {N_, nx_});
        }

        const bool cost = !Q.is_none() || !R.is_none() || !S.is_none();
        if (cost && (Q.is_none() || R.is_none() || S.is_none()))
            throw py::value_error("Q, R, and S must be updated together");
        if (cost) {
            Qp = fixed_array(Q, "Q", {N_, nx_, nx_});
            Rp = fixed_array(R, "R", {N_, nu_, nu_});
            Sp = fixed_array(S, "S", {N_, nu_, nx_});
            check_cost(Qp, Rp, Sp, N_, nx_, nu_);
        }

        const bool constraints = !D.is_none() || !C.is_none() || !d.is_none() || !c.is_none();
        if (constraints && (D.is_none() || C.is_none() || d.is_none() || c.is_none()))
            throw py::value_error("D, C, d, and c must be updated together");
        if (constraints) {
            Dp = ragged_matrix(D, N_, nx_, "D");
            Cp = ragged_matrix(C, N_, nu_, "C");
            dp = ragged_vector(d, N_, "d");
            cp = ragged_vector(c, N_, "c");
            same_rows(Dp, dp, "D", "d");
            same_rows(Cp, cp, "C", "c");
        }

        update_problem_data(
            &wrk_, const_cast<f64*>(x0p.data()),
            has_q ? const_cast<f64*>(qp.data()) : nullptr,
            has_r ? const_cast<f64*>(rp.data()) : nullptr,
            dynamics ? const_cast<f64*>(Ap.data()) : nullptr,
            dynamics ? const_cast<f64*>(Bp.data()) : nullptr,
            dynamics ? const_cast<f64*>(wp.data()) : nullptr,
            cost ? const_cast<f64*>(Qp.data()) : nullptr,
            cost ? const_cast<f64*>(Rp.data()) : nullptr,
            cost ? const_cast<f64*>(Sp.data()) : nullptr,
            constraints ? Dp.rows.data() : nullptr,
            constraints ? Cp.rows.data() : nullptr,
            constraints ? Dp.data() : nullptr,
            constraints ? Cp.data() : nullptr,
            constraints ? dp.data() : nullptr,
            constraints ? cp.data() : nullptr);
    }

private:
    workspace wrk_{};
    bool initialized_;
    u32 N_, nx_, nu_;
};

}  // namespace

PYBIND11_MODULE(daocp, m) {
    m.doc() = "Python interface for the OCP-structured dual active-set solver";
    m.attr("SOLVED") = py::int_(static_cast<int>(SOLVED));
    m.attr("INFEASIBLE") = py::int_(static_cast<int>(INFEASIBLE));
    m.attr("MAX_ITER") = py::int_(static_cast<int>(MAX_ITER));
    m.attr("ILL_CONDITIONED") = py::int_(static_cast<int>(ILL_CONDITIONED));

    py::class_<SolveInfo>(m, "SolveInfo")
        .def_readonly("status", &SolveInfo::status)
        .def_readonly("iters", &SolveInfo::iters)
        .def_property_readonly("iter", [](const SolveInfo& info) { return info.iters; })
        .def_readonly("solve_time", &SolveInfo::solve_time);

    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("x", &SolveResult::x)
        .def_readonly("u", &SolveResult::u)
        .def_readonly("info", &SolveResult::info);

    py::class_<OCPsolver>(m, "OCPsolver")
        .def(py::init<py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, py::object, u32, u32, u32, u32>(),
             py::arg("A"), py::arg("B"), py::arg("w"), py::arg("Q"), py::arg("R"),
             py::arg("S"), py::arg("q"), py::arg("r"), py::arg("D"), py::arg("C"),
             py::arg("d"), py::arg("c"), py::arg("x0"), py::arg("N"), py::arg("nx"),
             py::arg("nu"), py::arg("max_iter"))
        .def("solve", &OCPsolver::solve)
        .def("update", &OCPsolver::update,
             py::arg("x0"), py::arg("q") = py::none(), py::arg("r") = py::none(),
             py::arg("A") = py::none(), py::arg("B") = py::none(),
             py::arg("w") = py::none(), py::arg("Q") = py::none(),
             py::arg("R") = py::none(), py::arg("S") = py::none(),
             py::arg("D") = py::none(), py::arg("C") = py::none(),
             py::arg("d") = py::none(), py::arg("c") = py::none());
}
