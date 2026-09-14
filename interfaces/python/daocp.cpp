/*
 * Copyright (c) 2026 Alberto Zaupa
 * SPDX-License-Identifier: MIT
 * See LICENSE in the project root for license information.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <daocp.h>
}

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::string stage_name(const char* name, u32 stage) {
    return std::string(name) + " stage " + std::to_string(stage);
}

Array fixed_array(py::handle value, const char* name,
                  std::initializer_list<py::ssize_t> shape) {
    if (value.is_none())
        throw py::value_error(std::string(name) + " cannot be None");
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

Array horizon_array(py::handle value, const char* name, u32 stages,
                    std::initializer_list<py::ssize_t> stage_shape,
                    bool required) {
    std::vector<py::ssize_t> output_shape{static_cast<py::ssize_t>(stages)};
    output_shape.insert(output_shape.end(), stage_shape.begin(), stage_shape.end());
    py::ssize_t stage_size = 1;
    for (py::ssize_t size : stage_shape) stage_size *= size;

    if (value.is_none()) {
        if (required)
            throw py::value_error(std::string(name) + " cannot be None");
        Array out(output_shape);
        std::fill_n(out.mutable_data(), out.size(), 0.0);
        return out;
    }

    Array input = Array::ensure(value);
    if (!input)
        throw py::type_error(std::string(name) + " must be a real-valued array");
    const py::ssize_t stage_ndim = static_cast<py::ssize_t>(stage_shape.size());
    if (input.ndim() == stage_ndim + 1) {
        if (input.shape(0) != stages)
            throw py::value_error(std::string(name) + " must contain " +
                                  std::to_string(stages) + " stages");
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
                              " must be a single-stage array or a horizon array");
    py::ssize_t axis = 0;
    for (py::ssize_t size : stage_shape) {
        if (input.shape(axis) != size)
            throw py::value_error(std::string(name) + " has an inconsistent shape");
        ++axis;
    }
    Array out(output_shape);
    for (u32 stage = 0; stage < stages; ++stage)
        std::copy_n(input.data(), stage_size,
                    out.mutable_data() + static_cast<py::ssize_t>(stage) * stage_size);
    return out;
}

std::vector<std::vector<double>> bound_horizon(
        py::handle value, const char* name, u32 stages, u32 dimension,
        double trivial_value) {
    std::vector<std::vector<double>> out(
        stages, std::vector<double>(dimension, trivial_value));
    if (value.is_none()) return out;

    Array dense = Array::ensure(value);
    if (dense) {
        if (dense.ndim() == 1) {
            if (dense.shape(0) != dimension)
                throw py::value_error(std::string(name) + " has an inconsistent shape");
            for (u32 stage = 0; stage < stages; ++stage)
                std::copy_n(dense.data(), dimension, out[stage].begin());
            return out;
        }
        if (dense.ndim() == 2 && dense.shape(0) == stages &&
            dense.shape(1) == dimension) {
            for (u32 stage = 0; stage < stages; ++stage)
                std::copy_n(dense.data() + static_cast<size_t>(stage) * dimension,
                            dimension, out[stage].begin());
            return out;
        }
    }

    if (!py::isinstance<py::sequence>(value))
        throw py::type_error(std::string(name) + " must be a real-valued array");
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(sequence) != stages)
        throw py::value_error(std::string(name) + " must contain " +
                              std::to_string(stages) + " stages");
    for (u32 stage = 0; stage < stages; ++stage) {
        py::object item = sequence[stage];
        if (item.is_none()) continue;
        Array parsed = fixed_array(item, name, {static_cast<py::ssize_t>(dimension)});
        std::copy_n(parsed.data(), dimension, out[stage].begin());
    }
    return out;
}

void check_bound_pair(double lower, double upper, const char* lower_name,
                      const char* upper_name, u32 stage, u32 row) {
    if (std::isnan(lower) || std::isnan(upper))
        throw py::value_error(std::string(lower_name) + " and " + upper_name +
                              " must not contain NaN values");
    if (lower == kInfinity || upper == -kInfinity)
        throw py::value_error(stage_name(lower_name, stage) + " / " + upper_name +
                              " row " + std::to_string(row) +
                              " contains an invalid infinite bound");
    if (lower > upper)
        throw py::value_error(std::string(lower_name) + " must be less than or equal to " +
                              upper_name + " elementwise");
    if (lower == upper && !std::isfinite(lower))
        throw py::value_error(std::string(lower_name) + " and " + upper_name +
                              " cannot define an equality at infinity");
}

bool trivial_pair(double lower, double upper) {
    return lower == -kInfinity && upper == kInfinity;
}

struct GeneralStage {
    std::vector<double> Cu;
    std::vector<double> Cx;
    u32 rows = 0;
};

GeneralStage general_stage(py::handle value, u32 stage, u32 nu, u32 nx) {
    if (!py::isinstance<py::sequence>(value))
        throw py::type_error(stage_name("C", stage) + " must be [C_u, C_x]");
    py::sequence pair = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(pair) != 2)
        throw py::value_error(stage_name("C", stage) + " must contain [C_u, C_x]");

    py::object u_object = pair[0];
    py::object x_object = pair[1];
    Array Cu;
    Array Cx;
    const bool has_u = !u_object.is_none();
    const bool has_x = !x_object.is_none();
    u32 rows_u = 0;
    u32 rows_x = 0;
    if (has_u) {
        Cu = Array::ensure(u_object);
        if (!Cu || Cu.ndim() != 2 || Cu.shape(1) != nu)
            throw py::value_error(stage_name("C_u", stage) +
                                  " has an inconsistent shape");
        if (Cu.shape(0) > std::numeric_limits<u32>::max())
            throw py::value_error(stage_name("C_u", stage) + " has too many rows");
        rows_u = static_cast<u32>(Cu.shape(0));
    }
    if (has_x) {
        Cx = Array::ensure(x_object);
        if (!Cx || Cx.ndim() != 2 || Cx.shape(1) != nx)
            throw py::value_error(stage_name("C_x", stage) +
                                  " has an inconsistent shape");
        if (Cx.shape(0) > std::numeric_limits<u32>::max())
            throw py::value_error(stage_name("C_x", stage) + " has too many rows");
        rows_x = static_cast<u32>(Cx.shape(0));
    }
    if (has_u && has_x && rows_u != rows_x)
        throw py::value_error(stage_name("C_u and C_x", stage) +
                              " must have the same number of rows");

    GeneralStage out;
    out.rows = has_u ? rows_u : rows_x;
    out.Cu.assign(static_cast<size_t>(out.rows) * nu, 0.0);
    out.Cx.assign(static_cast<size_t>(out.rows) * nx, 0.0);
    if (has_u) std::copy_n(Cu.data(), Cu.size(), out.Cu.begin());
    if (has_x) std::copy_n(Cx.data(), Cx.size(), out.Cx.begin());
    return out;
}

std::vector<GeneralStage> general_horizon(py::handle value, u32 N, u32 nx, u32 nu) {
    std::vector<GeneralStage> out(N + 1);
    if (value.is_none()) return out;
    if (!py::isinstance<py::sequence>(value))
        throw py::type_error("C must be a sequence of [C_u[t], C_x[t]] pairs");
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(sequence) != N + 1)
        throw py::value_error("C must contain N + 1 stages");
    for (u32 stage = 0; stage <= N; ++stage)
        out[stage] = general_stage(sequence[stage], stage, stage < N ? nu : 0, nx);
    return out;
}

std::vector<double> general_bound_stage(
        py::handle value, const char* name, u32 stage, u32 stages,
        u32 rows, double trivial_value) {
    if (value.is_none()) return std::vector<double>(rows, trivial_value);
    Array dense = Array::ensure(value);
    if (dense) {
        if (dense.ndim() == 1) {
            if (dense.shape(0) != rows)
                throw py::value_error(stage_name(name, stage) +
                                      " has an inconsistent shape");
            return std::vector<double>(dense.data(), dense.data() + dense.size());
        }
        if (dense.ndim() == 2 && dense.shape(0) == stages) {
            if (dense.shape(1) != rows)
                throw py::value_error(stage_name(name, stage) +
                                      " has an inconsistent shape");
            const double* first = dense.data() + static_cast<size_t>(stage) * rows;
            return std::vector<double>(first, first + rows);
        }
    }
    if (!py::isinstance<py::sequence>(value))
        throw py::type_error(std::string(name) + " must be a real-valued array");
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(sequence) != stages)
        throw py::value_error(std::string(name) + " must contain N + 1 stages");
    py::object item = sequence[stage];
    if (item.is_none()) return std::vector<double>(rows, trivial_value);
    Array parsed = fixed_array(item, name, {static_cast<py::ssize_t>(rows)});
    return std::vector<double>(parsed.data(), parsed.data() + parsed.size());
}

struct StageData {
    std::vector<u32> idxbu;
    std::vector<u32> idxbx;
    std::vector<double> lbu;
    std::vector<double> ubu;
    std::vector<double> lbx;
    std::vector<double> ubx;
    std::vector<double> Cu;
    std::vector<double> Cx;
    std::vector<double> cl;
    std::vector<double> cu;
    std::vector<double> Du;
    std::vector<double> Dx;
    std::vector<double> d;
};

void append_equality(StageData& stage, const double* Du, u32 nu,
                     const double* Dx, u32 nx, double value) {
    if (nu) stage.Du.insert(stage.Du.end(), Du, Du + nu);
    stage.Dx.insert(stage.Dx.end(), Dx, Dx + nx);
    stage.d.push_back(value);
}

void append_unit_equality(StageData& stage, u32 nu, u32 nx,
                          bool input, u32 index, double value) {
    const size_t du_start = stage.Du.size();
    const size_t dx_start = stage.Dx.size();
    stage.Du.resize(du_start + nu, 0.0);
    stage.Dx.resize(dx_start + nx, 0.0);
    if (input)
        stage.Du[du_start + index] = 1.0;
    else
        stage.Dx[dx_start + index] = 1.0;
    stage.d.push_back(value);
}

void process_simple_bounds(StageData& stage,
                           const std::vector<double>& lower,
                           const std::vector<double>& upper,
                           bool input, u32 nu, u32 nx, u32 solver_stage,
                           const char* lower_name, const char* upper_name) {
    const u32 dimension = input ? nu : nx;
    for (u32 index = 0; index < dimension; ++index) {
        const double lb = lower[index];
        const double ub = upper[index];
        check_bound_pair(lb, ub, lower_name, upper_name, solver_stage, index);
        if (trivial_pair(lb, ub)) continue;
        if (lb == ub) {
            append_unit_equality(stage, nu, nx, input, index, lb);
        } else if (input) {
            stage.idxbu.push_back(index);
            stage.lbu.push_back(lb);
            stage.ubu.push_back(ub);
        } else {
            stage.idxbx.push_back(index);
            stage.lbx.push_back(lb);
            stage.ubx.push_back(ub);
        }
    }
}

bool symmetric(const double* matrix, u32 n) {
    double scale = 1.0;
    for (u32 i = 0; i < n * n; ++i) {
        if (!std::isfinite(matrix[i])) return false;
        scale = std::max(scale, std::abs(matrix[i]));
    }
    const double tolerance = 1e-10 * scale;
    for (u32 row = 0; row < n; ++row)
        for (u32 column = 0; column < row; ++column)
            if (std::abs(matrix[row * n + column] - matrix[column * n + row]) >
                tolerance)
                return false;
    return true;
}

bool cholesky_check(const double* matrix, u32 n, double regularization) {
    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    for (u32 row = 0; row < n; ++row) {
        for (u32 column = 0; column <= row; ++column) {
            double value = matrix[row * n + column];
            if (row == column) value += regularization;
            for (u32 k = 0; k < column; ++k)
                value -= L[row * n + k] * L[column * n + k];
            if (row == column) {
                if (!std::isfinite(value) || value <= 0.0) return false;
                L[row * n + column] = std::sqrt(value);
            } else {
                L[row * n + column] = value / L[column * n + column];
            }
        }
    }
    return true;
}

double psd_epsilon(const double* matrix, u32 n) {
    double scale = 1.0;
    for (u32 i = 0; i < n; ++i)
        scale = std::max(scale, std::abs(matrix[i * n + i]));
    return 1e-10 * scale;
}

void check_cost(const Array& Q, const Array& R, const Array& S,
                u32 N, u32 nx, u32 nu) {
    const u32 nv = nu + nx;
    std::vector<double> block(static_cast<size_t>(nv) * nv, 0.0);
    for (u32 stage = 0; stage < N; ++stage) {
        const double* Qt = Q.data() + static_cast<size_t>(stage) * nx * nx;
        const double* Rt = R.data() + static_cast<size_t>(stage) * nu * nu;
        const double* St = S.data() + static_cast<size_t>(stage) * nu * nx;
        if (!symmetric(Rt, nu) || !cholesky_check(Rt, nu, 0.0))
            throw py::value_error("R must be symmetric positive definite at every stage");
        if (!symmetric(Qt, nx) || !cholesky_check(Qt, nx, psd_epsilon(Qt, nx)))
            throw py::value_error("Q must be symmetric positive semidefinite at every stage");
        std::fill(block.begin(), block.end(), 0.0);
        for (u32 row = 0; row < nu; ++row) {
            for (u32 column = 0; column < nu; ++column)
                block[row * nv + column] = Rt[row * nu + column];
            for (u32 column = 0; column < nx; ++column) {
                block[row * nv + nu + column] = St[row * nx + column];
                block[(nu + column) * nv + row] = St[row * nx + column];
            }
        }
        for (u32 row = 0; row < nx; ++row)
            for (u32 column = 0; column < nx; ++column)
                block[(nu + row) * nv + nu + column] = Qt[row * nx + column];
        if (!cholesky_check(block.data(), nv, psd_epsilon(block.data(), nv)))
            throw py::value_error("[R S; S' Q] must be positive semidefinite at every stage");
    }
    const double* QN = Q.data() + static_cast<size_t>(N) * nx * nx;
    if (!symmetric(QN, nx) || !cholesky_check(QN, nx, psd_epsilon(QN, nx)))
        throw py::value_error("terminal Q must be symmetric positive semidefinite");
}

struct StageZeroShift {
    bool equality;
    u32 row;
    std::vector<double> Cx;
    double lower;
    double upper;
};

struct SolveInfo {
    std::string status;
    u32 iters;
    double solve_time;
};

const char* status_string(daocp_status status) {
    switch (status) {
        case DAOCP_SOLVED: return "SOLVED";
        case DAOCP_INFEASIBLE: return "INFEASIBLE";
        case DAOCP_MAX_ITER: return "MAX_ITER";
        case DAOCP_ILL_CONDITIONED: return "ILL_CONDITIONED";
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
    f64 setup_time;
    OCPsolver(py::object A, py::object B, py::object w,
              py::object Q, py::object R, py::object S,
              py::object q, py::object r,
              py::object lbx, py::object ubx,
              py::object lbu, py::object ubu,
              py::object C, py::object c_l, py::object c_u,
              py::object x0, u32 N, u32 nx, u32 nu, u32 max_iter,
              bool greedy, f64 pr_tol, f64 du_tol)
        : N_(N), nx_(nx), nu_(nu),
          nx_dims_(N + 1, nx), nu_dims_(N + 1, nu),
          nbu_dims_(N + 1, 0), nbx_dims_(N + 1, 0),
          ng_dims_(N + 1, 0), ne_dims_(N + 1, 0) {
        if (!N || !nx || !nu || !max_iter)
            throw py::value_error("N, nx, nu, and max_iter must be positive");
        if (pr_tol <= 0 || du_tol <= 0) 
            throw py::value_error("pr_tol and du_tol must be positive");
        nu_dims_[N] = 0;

        Array Ap = horizon_array(A, "A", N, {nx, nx}, true);
        Array Bp = horizon_array(B, "B", N, {nx, nu}, true);
        Array wp = horizon_array(w, "w", N, {nx}, false);
        Array Qp = horizon_array(Q, "Q", N + 1, {nx, nx}, false);
        Array Rp = horizon_array(R, "R", N, {nu, nu}, true);
        Array Sp = horizon_array(S, "S", N, {nu, nx}, false);
        Array qp = horizon_array(q, "q", N + 1, {nx}, false);
        Array rp = horizon_array(r, "r", N, {nu}, false);
        Array x0p = fixed_array(x0, "x0", {nx});
        check_cost(Qp, Rp, Sp, N, nx, nu);

        auto lbup = bound_horizon(lbu, "lbu", N, nu, -kInfinity);
        auto ubup = bound_horizon(ubu, "ubu", N, nu, kInfinity);
        auto lbxp = bound_horizon(lbx, "lbx", N, nx, -kInfinity);
        auto ubxp = bound_horizon(ubx, "ubx", N, nx, kInfinity);
        std::vector<GeneralStage> general = general_horizon(C, N, nx, nu);
        if (C.is_none() && (!c_l.is_none() || !c_u.is_none()))
            throw py::value_error("c_l and c_u require C");

        stages_.resize(N + 1);
        for (u32 stage = 0; stage < N; ++stage)
            process_simple_bounds(stages_[stage], lbup[stage], ubup[stage], true,
                                  nu, nx, stage, "lbu", "ubu");
        for (u32 horizon_index = 0; horizon_index < N; ++horizon_index) {
            const u32 stage = horizon_index + 1;
            process_simple_bounds(stages_[stage], lbxp[horizon_index], ubxp[horizon_index],
                                  false, stage < N ? nu : 0, nx, stage,
                                  "lbx", "ubx");
        }

        x0_.assign(x0p.data(), x0p.data() + x0p.size());
        for (u32 stage = 0; stage <= N; ++stage) {
            const u32 stage_nu = stage < N ? nu : 0;
            const GeneralStage& source = general[stage];
            std::vector<double> lower = general_bound_stage(
                c_l, "c_l", stage, N + 1, source.rows, -kInfinity);
            std::vector<double> upper = general_bound_stage(
                c_u, "c_u", stage, N + 1, source.rows, kInfinity);
            for (u32 row = 0; row < source.rows; ++row) {
                double row_lower = lower[row];
                double row_upper = upper[row];
                check_bound_pair(row_lower, row_upper, "c_l", "c_u", stage, row);
                if (trivial_pair(row_lower, row_upper)) continue;
                const double* Cu_row = stage_nu
                    ? source.Cu.data() + static_cast<size_t>(row) * stage_nu : nullptr;
                const double* Cx_row = source.Cx.data() + static_cast<size_t>(row) * nx;
                if (stage == 0) {
                    double shift = 0.0;
                    for (u32 column = 0; column < nx; ++column)
                        shift += Cx_row[column] * x0_[column];
                    row_lower -= shift;
                    row_upper -= shift;
                }
                if (row_lower == row_upper) {
                    const u32 equality_row = static_cast<u32>(stages_[stage].d.size());
                    std::vector<double> zero_x(nx, 0.0);
                    append_equality(stages_[stage], Cu_row, stage_nu,
                                    stage == 0 ? zero_x.data() : Cx_row, nx,
                                    row_lower);
                    if (stage == 0)
                        stage_zero_shifts_.push_back(
                            {true, equality_row,
                             std::vector<double>(Cx_row, Cx_row + nx),
                             lower[row], upper[row]});
                } else {
                    const u32 inequality_row = static_cast<u32>(stages_[stage].cl.size());
                    if (stage_nu)
                        stages_[stage].Cu.insert(stages_[stage].Cu.end(),
                                                 Cu_row, Cu_row + stage_nu);
                    if (stage == 0)
                        stages_[stage].Cx.insert(stages_[stage].Cx.end(), nx, 0.0);
                    else
                        stages_[stage].Cx.insert(stages_[stage].Cx.end(),
                                                 Cx_row, Cx_row + nx);
                    stages_[stage].cl.push_back(row_lower);
                    stages_[stage].cu.push_back(row_upper);
                    if (stage == 0)
                        stage_zero_shifts_.push_back(
                            {false, inequality_row,
                             std::vector<double>(Cx_row, Cx_row + nx),
                             lower[row], upper[row]});
                }
            }
        }

        for (u32 stage = 0; stage <= N; ++stage) {
            nbu_dims_[stage] = static_cast<u32>(stages_[stage].idxbu.size());
            nbx_dims_[stage] = static_cast<u32>(stages_[stage].idxbx.size());
            ng_dims_[stage] = static_cast<u32>(stages_[stage].cl.size());
            ne_dims_[stage] = static_cast<u32>(stages_[stage].d.size());
        }
        dims_ = {N, nx_dims_.data(), nu_dims_.data(), nbu_dims_.data(),
                 nbx_dims_.data(), ng_dims_.data(), ne_dims_.data()};
        qp_memory_.resize(daocp_qp_memsize(&dims_));
        daocp_qp_memory_assign(&dims_, &qp_, qp_memory_.data());
        populate_qp(Ap, Bp, wp, Qp, Rp, Sp, qp, rp);
        workspace_memory_.resize(daocp_workspace_memsize(&dims_));
        auto start = std::chrono::high_resolution_clock::now();
        daocp_workspace_memory_assign(&dims_, &qp_, workspace_memory_.data());
        auto end = std::chrono::high_resolution_clock::now();
        setup_time = std::chrono::duration<f64, std::micro>(end - start).count();
        solution_memory_.resize(daocp_sol_memsize(&dims_));
        daocp_sol_memory_assign(&dims_, &solution_, solution_memory_.data());

        daocp_args_set_default(&args_);
        args_.max_iter = max_iter;
        args_.selection = greedy ? DAOCP_SELECT_GREEDY : DAOCP_SELECT_MOST_VIOLATED;
        args_.primal_tol = pr_tol; 
        args_.dual_tol = du_tol;
        q_.assign(qp.data(), qp.data() + qp.size());
        r_.assign(rp.data(), rp.data() + rp.size());
        rebuild_linear_descriptors();
    }

    OCPsolver(const OCPsolver&) = delete;
    OCPsolver& operator=(const OCPsolver&) = delete;

    SolveResult solve() {
        const auto start = std::chrono::steady_clock::now();
        {
            py::gil_scoped_release release;
            daocp_solve(&args_, &qp_, workspace_memory_.data(), &solution_);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double micros =
            std::chrono::duration<double, std::micro>(stop - start).count();
        py::array_t<double> x({static_cast<py::ssize_t>(N_),
                               static_cast<py::ssize_t>(nx_)});
        py::array_t<double> u({static_cast<py::ssize_t>(N_),
                               static_cast<py::ssize_t>(nu_)});
        for (u32 stage = 0; stage < N_; ++stage)
            std::copy_n(solution_.ux[stage].pa, nu_,
                        u.mutable_data() + static_cast<size_t>(stage) * nu_);
        for (u32 stage = 1; stage <= N_; ++stage)
            std::copy_n(solution_.ux[stage].pa + nu_dims_[stage], nx_,
                        x.mutable_data() + static_cast<size_t>(stage - 1) * nx_);
        const daocp_status status = daocp_workspace_status(workspace_memory_.data());
        const u32 iterations = daocp_workspace_iterations(workspace_memory_.data());
        return {std::move(x), std::move(u),
                {status_string(status), iterations, micros}};
    }

    void update(py::object x0, py::object q, py::object r) {
        Array x0p = fixed_array(x0, "x0", {nx_});
        x0_.assign(x0p.data(), x0p.data() + x0p.size());
        if (!q.is_none()) {
            Array parsed = horizon_array(q, "q", N_ + 1, {nx_}, false);
            q_.assign(parsed.data(), parsed.data() + parsed.size());
        }
        if (!r.is_none()) {
            Array parsed = horizon_array(r, "r", N_, {nu_}, false);
            r_.assign(parsed.data(), parsed.data() + parsed.size());
        }
        rebuild_linear_descriptors();
        update_stage_zero_bounds();
        daocp_update_problem(workspace_memory_.data(), &qp_, x0_.data(),
                             linear_.data(), qp_.lbx, qp_.ubx,
                             qp_.lbu, qp_.ubu, qp_.cl, qp_.cu);
        const bool shifted_equality = std::any_of(
            stage_zero_shifts_.begin(), stage_zero_shifts_.end(),
            [](const StageZeroShift& shift) { return shift.equality; });
        if (shifted_equality)
            daocp_workspace_memory_assign(&dims_, &qp_, workspace_memory_.data());
    }

private:
    void populate_qp(const Array& A, const Array& B, const Array& w,
                     const Array& Q, const Array& R, const Array& S,
                     const Array& q, const Array& r) {
        std::copy(x0_.begin(), x0_.end(), qp_.x0);
        struct blasfeo_dvec vector_view{};
        for (u32 stage = 0; stage < N_; ++stage) {
            const double* At = A.data() + static_cast<size_t>(stage) * nx_ * nx_;
            const double* Bt = B.data() + static_cast<size_t>(stage) * nx_ * nu_;
            const double* wt = w.data() + static_cast<size_t>(stage) * nx_;
            blasfeo_dgese(nu_ + nx_ + 1, nx_, 0.0, qp_.BAwt + stage, 0, 0);
            blasfeo_pack_dmat(nu_, nx_, const_cast<double*>(Bt), nu_,
                              qp_.BAwt + stage, 0, 0);
            blasfeo_pack_dmat(nx_, nx_, const_cast<double*>(At), nx_,
                              qp_.BAwt + stage, nu_, 0);
            vector_view.pa = const_cast<double*>(wt);
            blasfeo_drowin(nx_, 1.0, &vector_view, 0, qp_.BAwt + stage,
                           nu_ + nx_, 0);
        }

        for (u32 stage = 0; stage <= N_; ++stage) {
            const u32 stage_nu = nu_dims_[stage];
            const u32 nv = stage_nu + nx_;
            const double* Qt = Q.data() + static_cast<size_t>(stage) * nx_ * nx_;
            const double* qt = q.data() + static_cast<size_t>(stage) * nx_;
            blasfeo_dgese(nv + 1, nv, 0.0, qp_.RSQrq + stage, 0, 0);
            if (stage < N_) {
                const double* Rt = R.data() + static_cast<size_t>(stage) * nu_ * nu_;
                const double* St = S.data() + static_cast<size_t>(stage) * nu_ * nx_;
                const double* rt = r.data() + static_cast<size_t>(stage) * nu_;
                blasfeo_pack_tran_dmat(nu_, nu_, const_cast<double*>(Rt), nu_,
                                       qp_.RSQrq + stage, 0, 0);
                blasfeo_pack_dmat(nx_, nu_, const_cast<double*>(St), nx_,
                                  qp_.RSQrq + stage, nu_, 0);
                vector_view.pa = const_cast<double*>(rt);
                blasfeo_drowin(nu_, 1.0, &vector_view, 0, qp_.RSQrq + stage,
                               nv, 0);
            }
            blasfeo_pack_tran_dmat(nx_, nx_, const_cast<double*>(Qt), nx_,
                                   qp_.RSQrq + stage, stage_nu, stage_nu);
            vector_view.pa = const_cast<double*>(qt);
            blasfeo_drowin(nx_, 1.0, &vector_view, 0, qp_.RSQrq + stage,
                           nv, stage_nu);

            const StageData& source = stages_[stage];
            if (stage < N_) {
                std::copy(source.Cu.begin(), source.Cu.end(), qp_.Cu[stage]);
                std::copy(source.Du.begin(), source.Du.end(), qp_.Du[stage]);
                std::copy(source.idxbu.begin(), source.idxbu.end(), qp_.idxbu[stage]);
                std::copy(source.lbu.begin(), source.lbu.end(), qp_.lbu[stage]);
                std::copy(source.ubu.begin(), source.ubu.end(), qp_.ubu[stage]);
            }
            if (stage > 0) {
                std::copy(source.Cx.begin(), source.Cx.end(), qp_.Cx[stage]);
                std::copy(source.idxbx.begin(), source.idxbx.end(), qp_.idxbx[stage]);
                std::copy(source.lbx.begin(), source.lbx.end(), qp_.lbx[stage]);
                std::copy(source.ubx.begin(), source.ubx.end(), qp_.ubx[stage]);
            }
            std::copy(source.Dx.begin(), source.Dx.end(), qp_.Dx[stage]);
            std::copy(source.cl.begin(), source.cl.end(), qp_.cl[stage]);
            std::copy(source.cu.begin(), source.cu.end(), qp_.cu[stage]);
            std::copy(source.d.begin(), source.d.end(), qp_.d[stage]);
        }
    }

    void rebuild_linear_descriptors() {
        linear_values_.clear();
        linear_.assign(N_ + 1, {});
        linear_values_.reserve(static_cast<size_t>(N_) * (nu_ + nx_) + nx_);
        for (u32 stage = 0; stage <= N_; ++stage) {
            const size_t offset = linear_values_.size();
            if (stage < N_)
                linear_values_.insert(linear_values_.end(),
                                      r_.begin() + static_cast<size_t>(stage) * nu_,
                                      r_.begin() + static_cast<size_t>(stage + 1) * nu_);
            linear_values_.insert(linear_values_.end(),
                                  q_.begin() + static_cast<size_t>(stage) * nx_,
                                  q_.begin() + static_cast<size_t>(stage + 1) * nx_);
            linear_[stage].pa = linear_values_.data() + offset;
            linear_[stage].m = static_cast<int>(nu_dims_[stage] + nx_);
        }
    }

    void update_stage_zero_bounds() {
        for (const StageZeroShift& shift : stage_zero_shifts_) {
            double value = 0.0;
            for (u32 column = 0; column < nx_; ++column)
                value += shift.Cx[column] * x0_[column];
            if (shift.equality) {
                qp_.d[0][shift.row] = shift.lower - value;
            } else {
                qp_.cl[0][shift.row] = shift.lower - value;
                qp_.cu[0][shift.row] = shift.upper - value;
            }
        }
    }

    u32 N_;
    u32 nx_;
    u32 nu_;
    std::vector<u32> nx_dims_;
    std::vector<u32> nu_dims_;
    std::vector<u32> nbu_dims_;
    std::vector<u32> nbx_dims_;
    std::vector<u32> ng_dims_;
    std::vector<u32> ne_dims_;
    daocp_dims dims_{};
    daocp_qp qp_{};
    daocp_sol solution_{};
    daocp_args args_{};
    std::vector<StageData> stages_;
    std::vector<StageZeroShift> stage_zero_shifts_;
    std::vector<unsigned char> qp_memory_;
    std::vector<unsigned char> workspace_memory_;
    std::vector<unsigned char> solution_memory_;
    std::vector<double> x0_;
    std::vector<double> q_;
    std::vector<double> r_;
    std::vector<double> linear_values_;
    std::vector<struct blasfeo_dvec> linear_;
};

}  // namespace

PYBIND11_MODULE(daocp, module) {
    module.doc() = "Python interface for the OCP-structured dual active-set solver";
    module.attr("SOLVED") = py::int_(static_cast<int>(DAOCP_SOLVED));
    module.attr("INFEASIBLE") = py::int_(static_cast<int>(DAOCP_INFEASIBLE));
    module.attr("MAX_ITER") = py::int_(static_cast<int>(DAOCP_MAX_ITER));
    module.attr("ILL_CONDITIONED") = py::int_(static_cast<int>(DAOCP_ILL_CONDITIONED));

    py::class_<SolveInfo>(module, "SolveInfo")
        .def_readonly("status", &SolveInfo::status)
        .def_readonly("iters", &SolveInfo::iters)
        .def_property_readonly("iter", [](const SolveInfo& info) { return info.iters; })
        .def_readonly("solve_time", &SolveInfo::solve_time);
    py::class_<SolveResult>(module, "SolveResult")
        .def_readonly("x", &SolveResult::x)
        .def_readonly("u", &SolveResult::u)
        .def_readonly("info", &SolveResult::info);
    py::class_<OCPsolver>(module, "OCPsolver")
        .def(py::init<py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, py::object, py::object, py::object,
                      py::object, py::object, py::object, py::object, py::object,
                      py::object, u32, u32, u32, u32, bool, f64, f64>(),
             py::arg("A"), py::arg("B"), py::arg("w"), py::arg("Q"),
             py::arg("R"), py::arg("S"), py::arg("q"), py::arg("r"),
             py::arg("lbx"), py::arg("ubx"), py::arg("lbu"), py::arg("ubu"),
             py::arg("C"), py::arg("c_l"), py::arg("c_u"), py::arg("x0"),
             py::arg("N"), py::arg("nx"), py::arg("nu"), py::arg("max_iter") = 1000,
             py::arg("greedy") = true, py::arg("pr_tol") = 1e-8,
             py::arg("du_tol") = 1e-8)
        .def_readwrite("setup_time", &OCPsolver::setup_time)
        .def("solve", &OCPsolver::solve)
        .def("update", &OCPsolver::update,
             py::arg("x0"), py::arg("q") = py::none(), py::arg("r") = py::none());
}
