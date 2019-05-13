#include "../data_processing/data.h"
#include <armadillo>

using namespace arma;

typedef struct {
    int M;
    int N;
    int K;
    Data Y;
    Data Y_test;
    Data Y_valid;
    double max_epochs;
} ModelParams;

class Model {

    ModelParams params;
    Mat<double> U;
    Mat<double> V;
    Col<double> del_U;
    Col<double> del_V;
    Mat<double> b_bin;
    Mat<double> b_u_tui;
    Col<double> b_i;
    Col<double> b_u;
    Col<double> alpha_u;
    Col<double> t_u;
    Col<double> c_u;
    Mat<double> f_ui;
    Mat<double> b_f_ui;
    vector<vector<int>> N_u;
    Col<int> N_u_size;
    Mat<double> Y;

    void user_date_avg();
    void user_frequency();

    Col<double> normalize_sum_y(int user);

    void update_y_vectors(
        int user,
        double del_common,
        Col<double>* Ui,
        int e);

    double devUser(
        int time,
        int user_avg
    );

    void grad_U(
        double del_common,
        Col<double> *Ui,
        Col<double> *Vj,
        int e
        );

    void grad_V(
        double del_common,
        Col<double> *Ui,
        Col<double> *Vj,
        Col<double> *y_norm,
        int e
        );

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
        double b_f_ui,
        Col<double> *Ui,
        Col<double> *Vj,
        Col<double> *y_norm
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
        double b_i,
        double c_u
    );

    double grad_b_bin(
        double del_common,
        double b_bin,
        double c_u
    );

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
        int K,
        string train_filename,
        string test_filename,
        string valid_filename,
        double max_epochs = 20
     );

    vector<double> predict();
    double trainErr();
    double validErr();
    void train();
    void save(char *file);
    ~Model();

};
