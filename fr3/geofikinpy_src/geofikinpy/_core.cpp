#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <array>
#include <vector>

#include "geofik/geofik.h"

namespace py = pybind11;

// ------------------ Utilities ------------------

template <typename T, size_t N>
std::array<T, N> to_std_array(const Eigen::Matrix<T, N, 1>& v) {
    std::array<T, N> arr{};
    for (size_t i = 0; i < N; i++) arr[i] = v(i);
    return arr;
}

// ------------------ FK & Jacobians ------------------

Eigen::Matrix4d franka_fk_eigen(const std::array<double,7>& q, char ee='E') {
    return franka_fk(q, ee);
}

std::array<std::array<double,6>,7> J_from_q_eigen(const std::array<double,7>& q, char ee='E') {
    return J_from_q(q, ee);
}

std::array<double,7> J_to_q_eigen(const std::array<std::array<double,6>,7>& Jt,
                                  const Eigen::Matrix3d& R, char ee='E') {
    std::array<std::array<double,3>,3> Rarr{};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) Rarr[i][j]=R(i,j);
    return J_to_q(Jt, Rarr, ee);
}

double swivel_from_q_eigen(const std::array<double,7>& q) {
    return franka_swivel(q);
}

// ------------------ IK Solvers ------------------

std::vector<std::array<double,7>> ik_q7_eigen(const Eigen::Vector3d& r,
                                              const Eigen::Matrix3d& ROE,
                                              double q7,
                                              double q1_sing=PI/2.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_ik_q7(rr,Rr,q7,qsols,q1_sing);

    std::vector<std::array<double,7>> out;
    for(unsigned i=0;i<n;i++) if(std::isfinite(qsols[i][0])) out.push_back(qsols[i]);
    return out;
}

std::vector<std::array<double,7>> ik_q4_eigen(const Eigen::Vector3d& r,
                                              const Eigen::Matrix3d& ROE,
                                              double q4,
                                              double q1_sing=PI/2.0,
                                              double q7_sing=0.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_ik_q4(rr,Rr,q4,qsols,q1_sing,q7_sing);

    std::vector<std::array<double,7>> out;
    for(unsigned i=0;i<n;i++) if(std::isfinite(qsols[i][0])) out.push_back(qsols[i]);
    return out;
}

std::vector<std::array<double,7>> ik_q6_eigen(const Eigen::Vector3d& r,
                                              const Eigen::Matrix3d& ROE,
                                              double q6,
                                              double q1_sing=PI/2.0,
                                              double q7_sing=0.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_ik_q6(rr,Rr,q6,qsols,q1_sing,q7_sing);

    std::vector<std::array<double,7>> out;
    for(unsigned i=0;i<n;i++) if(std::isfinite(qsols[i][0])) out.push_back(qsols[i]);
    return out;
}

std::vector<std::array<double,7>> ik_swivel_eigen(const Eigen::Vector3d& r,
                                                  const Eigen::Matrix3d& ROE,
                                                  double theta,
                                                  double q1_sing=PI/2.0,
                                                  unsigned int n_points=600) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_ik_swivel(rr,Rr,theta,qsols,q1_sing,n_points);

    std::vector<std::array<double,7>> out;
    for(unsigned i=0;i<n;i++) if(std::isfinite(qsols[i][0])) out.push_back(qsols[i]);
    return out;
}

// ------------------ Jacobian-augmented IK ------------------

using Jsol = std::pair<std::array<double,7>, Eigen::Matrix<double,6,7>>;

std::vector<Jsol> J_ik_q7_eigen(const Eigen::Vector3d& r,
                                const Eigen::Matrix3d& ROE,
                                double q7,
                                bool joint_angles=true,
                                char Jacobian_ee='E',
                                double q1_sing=PI/2.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<std::array<double,6>,7>,8> Jsols{};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_J_ik_q7(rr,Rr,q7,Jsols,qsols,joint_angles,Jacobian_ee,q1_sing);

    std::vector<Jsol> out;
    out.reserve(n);
    for(unsigned i=0;i<n;i++) {
        Eigen::Matrix<double,6,7> J;
        for(int c=0;c<7;c++) for(int r_=0;r_<6;r_++) J(r_,c)=Jsols[i][c][r_];
        out.emplace_back(qsols[i], J);
    }
    return out;
}

// (Similarly for q4, q6, swivel)
std::vector<Jsol> J_ik_q4_eigen(const Eigen::Vector3d& r,
                                const Eigen::Matrix3d& ROE,
                                double q4,
                                bool joint_angles=true,
                                char Jacobian_ee='E',
                                double q1_sing=PI/2.0,
                                double q7_sing=0.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<std::array<double,6>,7>,8> Jsols{};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_J_ik_q4(rr,Rr,q4,Jsols,qsols,joint_angles,Jacobian_ee,q1_sing,q7_sing);

    std::vector<Jsol> out;
    for(unsigned i=0;i<n;i++) {
        Eigen::Matrix<double,6,7> J;
        for(int c=0;c<7;c++) for(int r_=0;r_<6;r_++) J(r_,c)=Jsols[i][c][r_];
        out.emplace_back(qsols[i], J);
    }
    return out;
}

std::vector<Jsol> J_ik_q6_eigen(const Eigen::Vector3d& r,
                                const Eigen::Matrix3d& ROE,
                                double q6,
                                bool joint_angles=true,
                                char Jacobian_ee='E',
                                double q1_sing=PI/2.0,
                                double q7_sing=0.0) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<std::array<double,6>,7>,8> Jsols{};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_J_ik_q6(rr,Rr,q6,Jsols,qsols,joint_angles,Jacobian_ee,q1_sing,q7_sing);

    std::vector<Jsol> out;
    for(unsigned i=0;i<n;i++) {
        Eigen::Matrix<double,6,7> J;
        for(int c=0;c<7;c++) for(int r_=0;r_<6;r_++) J(r_,c)=Jsols[i][c][r_];
        out.emplace_back(qsols[i], J);
    }
    return out;
}

std::vector<Jsol> J_ik_swivel_eigen(const Eigen::Vector3d& r,
                                    const Eigen::Matrix3d& ROE,
                                    double theta,
                                    bool joint_angles=true,
                                    char Jacobian_ee='E',
                                    double q1_sing=PI/2.0,
                                    unsigned int n_points=600) {
    std::array<double,3> rr{r(0),r(1),r(2)};
    std::array<double,9> Rr{ROE(0,0),ROE(0,1),ROE(0,2),
                            ROE(1,0),ROE(1,1),ROE(1,2),
                            ROE(2,0),ROE(2,1),ROE(2,2)};
    std::array<std::array<std::array<double,6>,7>,8> Jsols{};
    std::array<std::array<double,7>,8> qsols{};
    unsigned int n = franka_J_ik_swivel(rr,Rr,theta,Jsols,qsols,joint_angles,Jacobian_ee,q1_sing,n_points);

    std::vector<Jsol> out;
    for(unsigned i=0;i<n;i++) {
        Eigen::Matrix<double,6,7> J;
        for(int c=0;c<7;c++) for(int r_=0;r_<6;r_++) J(r_,c)=Jsols[i][c][r_];
        out.emplace_back(qsols[i], J);
    }
    return out;
}

// ------------------ Module ------------------

PYBIND11_MODULE(_core, m) {
    m.doc() = "geofikinpy: GeoFIK full binding";

    // FK/J
    m.def("franka_fk", &franka_fk_eigen, py::arg("q"), py::arg("ee")='E');
    m.def("J_from_q", &J_from_q_eigen, py::arg("q"), py::arg("ee")='E');
    m.def("J_to_q", &J_to_q_eigen, py::arg("Jt"), py::arg("R"), py::arg("ee")='E');
    m.def("swivel_from_q", &swivel_from_q_eigen);

    // IK families
    m.def("ik_q7", &ik_q7_eigen, py::arg("r"), py::arg("ROE"), py::arg("q7"), py::arg("q1_sing")=PI/2.0);
    m.def("ik_q4", &ik_q4_eigen, py::arg("r"), py::arg("ROE"), py::arg("q4"), py::arg("q1_sing")=PI/2.0, py::arg("q7_sing")=0.0);
    m.def("ik_q6", &ik_q6_eigen, py::arg("r"), py::arg("ROE"), py::arg("q6"), py::arg("q1_sing")=PI/2.0, py::arg("q7_sing")=0.0);
    m.def("ik_swivel", &ik_swivel_eigen, py::arg("r"), py::arg("ROE"), py::arg("theta"), py::arg("q1_sing")=PI/2.0, py::arg("n_points")=600);

    // IK+Jacobian families
    m.def("J_ik_q7", &J_ik_q7_eigen, py::arg("r"), py::arg("ROE"), py::arg("q7"),
          py::arg("joint_angles")=true, py::arg("Jacobian_ee")='E', py::arg("q1_sing")=PI/2.0);
    m.def("J_ik_q4", &J_ik_q4_eigen, py::arg("r"), py::arg("ROE"), py::arg("q4"),
          py::arg("joint_angles")=true, py::arg("Jacobian_ee")='E', py::arg("q1_sing")=PI/2.0, py::arg("q7_sing")=0.0);
    m.def("J_ik_q6", &J_ik_q6_eigen, py::arg("r"), py::arg("ROE"), py::arg("q6"),
          py::arg("joint_angles")=true, py::arg("Jacobian_ee")='E', py::arg("q1_sing")=PI/2.0, py::arg("q7_sing")=0.0);
    m.def("J_ik_swivel", &J_ik_swivel_eigen, py::arg("r"), py::arg("ROE"), py::arg("theta"),
          py::arg("joint_angles")=true, py::arg("Jacobian_ee")='E', py::arg("q1_sing")=PI/2.0, py::arg("n_points")=600);
}
