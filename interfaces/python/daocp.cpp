#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
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

Array horizon_array(py::handle value, const char* name, u32 N,
                    std::initializer_list<py::ssize_t> stage_shape) {
    Array input = Array::ensure(value);
    if (!input)
        throw py::type_error(std::string(name) + " must be a real-valued array");

    const py::ssize_t stage_ndim = static_cast<py::ssize_t>(stage_shape.size());
    if (input.ndim() == stage_ndim + 1) {
        if (input.shape(0) != N)
            throw py::value_error(std::string(name) + " must contain N stages");
        py::ssize_t axis = 1;
        for (py::ssize_t size : stage_shape) {
            if (input.shape(axis) != size)
                throw py::value_error(std::string(name) + " has an inconsistent shape");
            ++axis;
        }
        return input;
    }

    if (input.ndim() != stage_ndim)
        throw py::value_error(std::string(name) +
                              " must be a single-stage array or contain N stages");
    py::ssize_t axis = 0;
    py::ssize_t stage_size = 1;
    std::vector<py::ssize_t> horizon_shape{static_cast<py::ssize_t>(N)};
    for (py::ssize_t size : stage_shape) {
        if (input.shape(axis) != size)
            throw py::value_error(std::string(name) + " has an inconsistent shape");
        horizon_shape.push_back(size);
        stage_size *= size;
        ++axis;
    }

    Array out(horizon_shape);
    for (u32 t = 0; t < N; ++t)
        std::copy_n(input.data(), stage_size,
                    out.mutable_data() + static_cast<py::ssize_t>(t) * stage_size);
    return out;
}

struct Ragged {
    Array dense;
    std::vector<double> values;
    std::vector<u32> rows;
    bool dense_backed = false;
    double empty_value = 0.0;

    f64* data() {
        if (dense_backed && dense.size()) return const_cast<f64*>(dense.data());
        return values.empty() ? &empty_value : values.data();
    }
};

Ragged empty_ragged(u32 N) {
    Ragged out;
    out.rows.assign(N, 0);
    return out;
}

Ragged ragged_matrix(py::handle value, u32 N, u32 columns, const char* name) {
    Ragged out;
    out.rows.reserve(N);

    if (py::isinstance<py::array>(value)) {
        out.dense = Array::ensure(value);
        if (!out.dense)
            throw py::type_error(std::string(name) + " must be real-valued");

        if (out.dense.ndim() == 2) {
            if (out.dense.shape(1) != columns)
                throw py::value_error(std::string(name) + " has an inconsistent shape");
            if (out.dense.shape(0) > std::numeric_limits<u32>::max())
                throw py::value_error(std::string(name) + " has too many rows");
            const u32 rows = static_cast<u32>(out.dense.shape(0));
            out.rows.assign(N, rows);
            out.values.reserve(static_cast<size_t>(N) * out.dense.size());
            for (u32 t = 0; t < N; ++t)
                out.values.insert(out.values.end(), out.dense.data(),
                                  out.dense.data() + out.dense.size());
            return out;
        }

        if (out.dense.ndim() != 3 || out.dense.shape(0) != N ||
            out.dense.shape(2) != columns)
            throw py::value_error(std::string(name) +
                                  " must have shape (m, n) or (N, m, n)");
        if (out.dense.shape(1) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " has too many rows");
        out.dense_backed = true;
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
        if (!out.dense)
            throw py::type_error(std::string(name) + " must be real-valued");

        if (out.dense.ndim() == 1) {
            if (out.dense.shape(0) > std::numeric_limits<u32>::max())
                throw py::value_error(std::string(name) + " is too long");
            const u32 rows = static_cast<u32>(out.dense.shape(0));
            out.rows.assign(N, rows);
            out.values.reserve(static_cast<size_t>(N) * out.dense.size());
            for (u32 t = 0; t < N; ++t)
                out.values.insert(out.values.end(), out.dense.data(),
                                  out.dense.data() + out.dense.size());
            return out;
        }

        if (out.dense.ndim() != 2 || out.dense.shape(0) != N)
            throw py::value_error(std::string(name) + " must have shape (m,) or (N, m)");
        if (out.dense.shape(1) > std::numeric_limits<u32>::max())
            throw py::value_error(std::string(name) + " is too long");
        out.dense_backed = true;
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

void optional_constraint_pair(py::handle matrix, py::handle bounds,
                              u32 N, u32 columns,
                              const char* matrix_name, const char* bounds_name,
                              Ragged& parsed_matrix, Ragged& parsed_bounds) {
    if (matrix.is_none() && bounds.is_none()) {
        parsed_matrix = empty_ragged(N);
        parsed_bounds = empty_ragged(N);
        return;
    }
    if (matrix.is_none() || bounds.is_none())
        throw py::value_error(std::string(matrix_name) + " and " + bounds_name +
                              " must either both be provided or both be None");
    parsed_matrix = ragged_matrix(matrix, N, columns, matrix_name);
    parsed_bounds = ragged_vector(bounds, N, bounds_name);
    same_rows(parsed_matrix, parsed_bounds, matrix_name, bounds_name);
}

void check_bound_order(Ragged& lower, Ragged& upper,
                       const char* lower_name, const char* upper_name) {
    const size_t count = std::accumulate(
        lower.rows.begin(), lower.rows.end(), static_cast<size_t>(0));
    const f64* lower_data = lower.data();
    const f64* upper_data = upper.data();
    for (size_t i = 0; i < count; ++i) {
        if (std::isnan(lower_data[i]) || std::isnan(upper_data[i]))
            throw py::value_error(std::string(lower_name) + " and " + upper_name +
                                  " must not contain NaN values");
        if (lower_data[i] > upper_data[i])
            throw py::value_error(std::string(lower_name) + " must be less than or equal to " +
                                  upper_name + " elementwise");
    }
}

void optional_two_sided_constraints(
        py::handle matrix, py::handle lower, py::handle upper,
        u32 N, u32 columns, const char* matrix_name,
        const char* lower_name, const char* upper_name,
        Ragged& parsed_matrix, Ragged& parsed_lower, Ragged& parsed_upper) {
    if (matrix.is_none() && lower.is_none() && upper.is_none()) {
        parsed_matrix = empty_ragged(N);
        parsed_lower = empty_ragged(N);
        parsed_upper = empty_ragged(N);
        return;
    }
    if (matrix.is_none() || lower.is_none() || upper.is_none())
        throw py::value_error(std::string(matrix_name) + ", " + lower_name + ", and " +
                              upper_name + " must either all be provided or all be None");

    parsed_matrix = ragged_matrix(matrix, N, columns, matrix_name);
    parsed_lower = ragged_vector(lower, N, lower_name);
    parsed_upper = ragged_vector(upper, N, upper_name);
    same_rows(parsed_matrix, parsed_lower, matrix_name, lower_name);
    same_rows(parsed_matrix, parsed_upper, matrix_name, upper_name);
    check_bound_order(parsed_lower, parsed_upper, lower_name, upper_name);
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
    std::string status;
    u32 iters;
    double solve_time;
};

const char* status_string(u32 status) {
    switch (status) {
        case SOLVED: return "SOLVED";
        case INFEASIBLE: return "INFEASIBLE";
        case MAX_ITER: return "MAX_ITER";
        case ILL_CONDITIONED: return "ILL_CONDITIONED";
        default: return "UNKNOWN";
    }
}

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
              py::object C, py::object du, py::object dl,
              py::object cu, py::object cl,
              py::object Deq, py::object Ceq,
              py::object deq, py::object ceq,
              py::object x0, u32 N, u32 nx, u32 nu, u32 max_iter,
              bool greedy)
        : initialized_(false), N_(N), nx_(nx), nu_(nu) {
        if (!N || !nx || !nu || !max_iter)
            throw py::value_error("N, nx, nu, and max_iter must be positive");

        Array Ap = horizon_array(A, "A", N, {nx, nx});
        Array Bp = horizon_array(B, "B", N, {nx, nu});
        Array wp = horizon_array(w, "w", N, {nx});
        Array Qp = horizon_array(Q, "Q", N, {nx, nx});
        Array Rp = horizon_array(R, "R", N, {nu, nu});
        Array Sp = horizon_array(S, "S", N, {nu, nx});
        Array qp = horizon_array(q, "q", N, {nx});
        Array rp = horizon_array(r, "r", N, {nu});
        Array x0p = fixed_array(x0, "x0", {nx});
        Ragged Dp, Cp, dup, dlp, cup, clp, Deqp, Ceqp, deqp, ceqp;
        optional_two_sided_constraints(
            D, dl, du, N, nx, "D", "dl", "du", Dp, dlp, dup);
        optional_two_sided_constraints(
            C, cl, cu, N, nu, "C", "cl", "cu", Cp, clp, cup);
        optional_constraint_pair(Deq, deq, N, nx, "Deq", "deq", Deqp, deqp);
        optional_constraint_pair(Ceq, ceq, N, nu, "Ceq", "ceq", Ceqp, ceqp);
        check_cost(Qp, Rp, Sp, N, nx, nu);

        workspace_init(&wrk_, const_cast<f64*>(Ap.data()), const_cast<f64*>(Bp.data()),
                       const_cast<f64*>(wp.data()), const_cast<f64*>(Qp.data()),
                       const_cast<f64*>(Rp.data()), const_cast<f64*>(Sp.data()),
                       const_cast<f64*>(qp.data()), const_cast<f64*>(rp.data()),
                       Dp.data(), Cp.data(), dup.data(), dlp.data(), cup.data(), clp.data(),
                       Deqp.data(), Ceqp.data(), deqp.data(), ceqp.data(),
                       const_cast<f64*>(x0p.data()), N, nx, nu,
                       Dp.rows.data(), Cp.rows.data(),
                       Deqp.rows.data(), Ceqp.rows.data(), max_iter, greedy);
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
            py::array_t<double>({N_, nx_}, {sizeof(double) * nx_, sizeof(double)},
                                wrk_.x + nx_, owner),
            py::array_t<double>({N_, nu_}, {sizeof(double) * nu_, sizeof(double)},
                                wrk_.u, owner),
            {status_string(wrk_.return_status), iters, micros}
        };
        return result;
    }

    void update(py::object x0, py::object q, py::object r) {
        Array x0p = fixed_array(x0, "x0", {nx_});
        Array qp, rp;

        const bool has_q = !q.is_none();
        const bool has_r = !r.is_none();
        if (has_q) qp = horizon_array(q, "q", N_, {nx_});
        if (has_r) rp = horizon_array(r, "r", N_, {nu_});

        ::update_problem_data(
            &wrk_, const_cast<f64*>(x0p.data()),
            has_q ? const_cast<f64*>(qp.data()) : nullptr,
            has_r ? const_cast<f64*>(rp.data()) : nullptr);
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
                      py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, py::object, py::object,
                      u32, u32, u32, u32, bool>(),
             py::arg("A"), py::arg("B"), py::arg("w"), py::arg("Q"), py::arg("R"),
             py::arg("S"), py::arg("q"), py::arg("r"), py::arg("D"), py::arg("C"),
             py::arg("du"), py::arg("dl"), py::arg("cu"), py::arg("cl"),
             py::arg("Deq"), py::arg("Ceq"),
             py::arg("deq"), py::arg("ceq"), py::arg("x0"), py::arg("N"),
             py::arg("nx"), py::arg("nu"), py::arg("max_iter"),
             py::arg("greedy") = true)
        .def("solve", &OCPsolver::solve)
        .def("update", &OCPsolver::update,
             py::arg("x0"), py::arg("q") = py::none(), py::arg("r") = py::none());
}
