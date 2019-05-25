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
    Col<double> b_i;
    Col<double> b_u;
    vector<vector<int>> N_u;
    Col<int> N_u_size;
    Mat<double> Y;
    Mat<double> Y_norm;
    Mat<int> Ratings;
    vector<vector<int>> Ratings_vec;

    void movies_per_user();

    //void initialize_y_norm();

    //void update_y_norm(int user);
    void compute_y_norm(int user);

    void update_y_vectors(
        int user,
        Col<double>* Vj,
        int e);


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
        double b_i,
        Col<double> *Ui,
        Col<double> *Vj,
        Col<double> *y_norm
    );

    double grad_b_u(
        double del_common,
        double b_u);


    double grad_b_i(
        double del_common,
        double b_i
    );


    public:

    Model(
        int M,
        int N,
        int K,
        string train_filename,
        string test_filename,
        string valid_filename,
        double max_epochs = 100
     );

    vector<double> predict();
    vector<double> predict_valid();
    double trainErr();
    double validErr();
    void train();
    void save(char *file);
    ~Model();

};
