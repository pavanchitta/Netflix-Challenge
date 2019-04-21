#include "netflix_data.h"
#include <armadillo>

using namespace arma;

typedef struct {
    int M;
    int N;
    NetflixData Y;
    double eps;
    double max_epochs;
} ModelParams;

class Model {
    ModelParams params;
    Mat<double> b_bin;
    Col<double> b_i;
    Col<double> b_u;
    Col<double> alpha_u;

    void userAvg(
            NetflixData Y);

    double grad_b_u(
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin);

    double grad_alpha_u(
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin);

    double grad_b_i(
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin);

    double grad_b_bin(
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin);

    public:
    Model(
        int M,
        int N,
        NetflixData Y,
        double eps = 0.01,
        double max_epochs = 200
     );

    double trainErr();
    void train();
    double devUser(int time, int user_avg);
    ~Model();
};
