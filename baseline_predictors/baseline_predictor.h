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
    Mat<double> b_u_tui;
    Col<double> b_i;
    Col<double> b_u;
    Col<double> alpha_u;
    Col<double> t_u;
    Col<double> c_u;
    Mat<double> f_ui;
    Mat<double> b_f_ui;

    void user_date_avg();
    void user_frequency();

    double grad_common(
            int user,
            int rating,
            double b_u,
            double alpha_u,
            int time,
            double b_i,
            double b_bin,
            double b_u_tui,
            double c_u,
            double b_f_ui
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

    double grad_b_u_tui(
        double del_common,
        double b_u_tui
    );

    double grad_c_u(
        double del_common,
        double c_u,
        double b_i,
        double b_bin
    );

    double grad_b_f_ui(
        double del_common,
        double b_f_ui
    );

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
