#include "xtensor.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

const double gamma = 1.4;
const double theta = gamma - 1;
unsigned int nx, ny;
double minX, minY, maxX, maxY;
double CFL = 0.1;

typedef xt::xtensor_fixed<double, xt::xshape<4>> State;

State fflux(State consvec) {
    double q1 = consvec(0), q2 = consvec(1), q3 = consvec(2), q4 = consvec(3);
    double p = theta*(q4 - 0.5*(q2*q2 + q3*q3)/q1);
    State f;
    f(0) = q2;
    f(1) = q2*q2/q1 + p;
    f(2) = q2*q3/q1;
    f(3) = q2*(q4 + p)/q1;
    return f;
}

State gflux(State consvec) {
    double q1 = consvec(0), q2 = consvec(1), q3 = consvec(2), q4 = consvec(3);
    double p = theta*(q4 - 0.5*(q2*q2 + q3*q3)/q1);
    State g;
    g(0) = q3;
    g(1) = q2*q3/q1;
    g(2) = q3*q3/q1 + p;
    g(3) = q3*(q4 + p)/q1;
    return g;
}

State primitive(State consvec) {
    double q1 = consvec(0), q2 = consvec(1), q3 = consvec(2), q4 = consvec(3);
    State prim;
    prim(0) = q1;
    prim(1) = q2/q1;
    prim(2) = q3/q1;
    prim(3) = theta*(q4 - 0.5*(q2*q2 + q3*q3)/q1);
    return prim;
}

State conservative(State primvec) {
    double rho = primvec(0), vx = primvec(1), vy = primvec(2), p = primvec(3);
    State cons;
    cons(0) = rho;
    cons(1) = rho*vx;
    cons(2) = rho*vy;
    cons(3) = p/theta + 0.5*rho*(vx*vx + vy*vy);
    return cons;
}

void LFfflux(xt::xtensor<double, 3> constens, xt::xtensor<double, 3> ffluxtens, xt::xtensor<double, 3> &LFffluxtens, double invstep) {
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            xt::view(LFffluxtens, i, j) = 0.5f * (xt::view(ffluxtens, i+1, j) - xt::view(ffluxtens, i-1, j) + invstep * (2 * xt::view(constens, i, j) - xt::view(constens, i+1, j) - xt::view(constens, i-1, j)));
        }
    }
    return;
}

void LFgflux(xt::xtensor<double, 3> constens, xt::xtensor<double, 3> gfluxtens, xt::xtensor<double, 3> &LFgfluxtens, double invstep) {
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            xt::view(LFgfluxtens, i, j) = 0.5f * (xt::view(gfluxtens, i, j+1) - xt::view(gfluxtens, i, j-1) + invstep * (2 * xt::view(constens, i, j) - xt::view(constens, i, j+1) - xt::view(constens, i, j-1)));
        }
    }
    return;
}

void RIfflux(xt::xtensor<double, 3> constens, xt::xtensor<double, 3> ffluxtens, xt::xtensor<double, 3> &RIffluxtens, double step) {
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            xt::view(RIffluxtens, i, j) = fflux(0.5f * (xt::view(constens, i, j) + xt::view(constens, i+1, j) + step*(xt::view(ffluxtens, i, j) - xt::view(ffluxtens, i+1, j)))) - fflux(0.5f * (xt::view(constens, i-1, j) + xt::view(constens, i, j) + step*(xt::view(ffluxtens, i-1, j) - xt::view(ffluxtens, i, j))));
        }
    }
    return;
}

void RIgflux(xt::xtensor<double, 3> constens, xt::xtensor<double, 3> gfluxtens, xt::xtensor<double, 3> &RIgfluxtens, double step) {
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
            xt::view(RIgfluxtens, i, j) = gflux(0.5f * (xt::view(constens, i, j) + xt::view(constens, i, j+1) + step*(xt::view(gfluxtens, i, j) - xt::view(gfluxtens, i, j+1)))) - gflux(0.5f * (xt::view(constens, i, j-1) + xt::view(constens, i, j) + step*(xt::view(gfluxtens, i, j-1) - xt::view(gfluxtens, i, j))));
        }
    }
    return;
}

double timestep(xt::xtensor<double, 3> primtens, double dd) {
    xt::xtensor<double, 1> a = xt::zeros<double>({nx*ny});
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            State pij = xt::view(primtens, i, j);
            double vij = sqrt(pij(1)*pij(1) + pij(2)*pij(2));
            double csij = sqrt(gamma * pij(3)/pij(0));
            a(i*j) = vij + csij;
        }
    }
    double amax = xt::amax(a)(0);
    return dd/amax;
}

int main() {

    State ambient = {1., 0., 0., 1.};
    State bubble = {0.1, 0., 0., 1.};
    double v_shock = 2.95;
    double rhov_s = ambient(0)*v_shock;
    double rhov_s2 = rhov_s*rhov_s;
    double s1, s2, s4;
    s4 = (2*rhov_s2 - theta*ambient(3))/(gamma + 1);
    s1 = rhov_s2/(rhov_s2 + ambient(3) - s4);
    s2 = (s4 - ambient(3))/(ambient(0) * v_shock) + ambient(1);
    State shock = {s1, s2, 0., s4};

    maxX = 1.6;
    minX = 0.;
    maxY = 1.;
    minY = 0.;

    std::cout << "Enter number of cells: ";
    std::cin >> nx;

    double dx = (maxX - minX)/nx;
    double dy = dx;

    clock_t start = clock();

    xt::xtensor<double, 1> xvec = xt::arange(minX, maxX, dx);
    xt::xtensor<double, 1> yvec = xt::arange(minY, maxY, dy);

    ny = yvec.size();

    xt::xarray<int>::shape_type sh = {nx, ny, 4};
    xt::xtensor<double, 3> primtens, constens, ffluxtens, gfluxtens, RIffluxtens, LFffluxtens, RIgfluxtens, LFgfluxtens;
    primtens = constens = ffluxtens = gfluxtens = RIffluxtens = LFffluxtens = RIgfluxtens = LFgfluxtens = xt::zeros<double>(sh);

    State consshock = conservative(shock), consbubble = conservative(bubble), consambient = conservative(ambient);
    State consfflux = fflux(consshock), consgflux = gflux(consshock), bubblefflux = fflux(consbubble), bubblegflux = gflux(consbubble);
    State ambientfflux = fflux(consambient), ambientgflux = gflux(consambient);

    unsigned int i, j;
    for (i = 0; i < nx; i++) {
        double x = xvec(i);
        if (x < 0.1) {
            xt::view(primtens, i) = shock;
            xt::view(constens, i) = consshock;
            xt::view(ffluxtens, i) = consfflux;
            xt::view(gfluxtens, i) = consgflux;
        }
        else {
            for (j = 0; j < ny; j++) {
                double y = yvec(j);
                double cond = sqrt(pow(x - 0.4, 2) + pow(y - 0.5, 2));
                if (cond < 0.2) {
                    xt::view(primtens, i, j) = bubble;
                    xt::view(constens, i, j) = consbubble;
                    xt::view(ffluxtens, i, j) = bubblefflux;
                    xt::view(gfluxtens, i, j) = bubblegflux;
                }
                else {
                    xt::view(primtens, i, j) = ambient;
                    xt::view(constens, i, j) = consambient;
                    xt::view(ffluxtens, i, j) = ambientfflux;
                    xt::view(gfluxtens, i, j) = ambientgflux;
                }
            }
        }
    }

    double step, dt;
    double t = 0, T = 0.3;
    int nsteps = 0;
    while (t < T) {
        nsteps++;
        dt = min(CFL*timestep(primtens, dx), T-t);
        t += dt;
        std::cout << "t = " << t << ", Step = " << nsteps << std::endl;
        step = dt/dx;
        RIfflux(constens, ffluxtens, RIffluxtens, step);
        RIgflux(constens, gfluxtens, RIgfluxtens, step);
        LFfflux(constens, ffluxtens, LFffluxtens, 1./step);
        LFgflux(constens, gfluxtens, LFgfluxtens, 1./step);
        constens -= 0.5 * step*(RIffluxtens + LFffluxtens + RIgfluxtens + LFgfluxtens);
        xt::view(constens, 0) = xt::view(constens, 1);
        xt::view(constens, nx-1) = xt::view(constens, nx-2);
        xt::view(constens, xt::all(), 0) = xt::view(constens, xt::all(), 1);
        xt::view(constens, xt::all(), ny-1) = xt::view(constens, xt::all(), ny-2);
        // std::cout << constens << std::endl;
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                xt::view(primtens, i, j) = primitive(xt::view(constens, i, j));
            }
        }
        // std::cout << primtens << std::endl;
        if (xt::allclose(t, T) == 0) {
            for (i = 0; i < nx; i++) {
                for (j = 0; j < ny; j++) {
                    xt::view(ffluxtens, i, j) = fflux(xt::view(constens, i, j));
                    xt::view(gfluxtens, i, j) = gflux(xt::view(constens, i, j));
                }
            }
        }
    }
    clock_t end = clock();
    std::cout << "It took " << (end - start) / (double)CLOCKS_PER_SEC << " seconds" << endl;
    std::cout << "Number of steps = " << nsteps << std::endl;
    std::cout << "Number of y cells = " << ny << std::endl;

    std::ofstream fp;
    fp.open("EulerSolution2D.out");
    State prims;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            prims = xt::view(primtens, i, j);
            fp << xvec(i) << " " << yvec(j) << " " << prims(0) << " " << prims(1) << " " << prims(2) << " " << prims(3) << std::endl;
        }
    }
    return 0;
}
