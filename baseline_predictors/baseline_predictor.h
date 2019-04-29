#include "../data_processing/data.h"
#include <armadillo>

using namespace arma;

typedef struct {
    int M;
    int N;
    Data Y;
    double eps;
    double max_epochs;
} ModelParams;

class Model {
    ModelParams params;
    Mat<double> b_bin;
    Col<double> b_i;
    Col<double> b_u;
    Col<double> alpha_u;
    Col<double> t_u;

    void user_date_avg();

    double grad_common(
            int user,
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin
    );

    double grad_b_u(
            double del_common,
            double b_u);

    double grad_alpha_u(
            double del_common,
            int user,
            int time,
            double alpha_u);

    double grad_b_i(
            double del_common,
            double b_i);

    double grad_b_bin(
            double del_common,
            double b_bin);

    public:
    Model(
        int M,
        int N,
        string filename,
        double eps = 0.01,
        double max_epochs = 40
     );

    double trainErr();
    void train();
    double devUser(int time, int user_avg);
    ~Model();
};
