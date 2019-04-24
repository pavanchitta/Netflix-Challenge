#include "../data_processing/data.h"
#include <armadillo>

using namespace arma;

typedef struct {
    int M;
    int N;
    int K;
    double eta;
    double reg;
    Data Y;
    Data Y_test;
    Data Y_valid;
    double mu;
    double eps;
    double max_epochs;
} ModelParams;

class Model {
    ModelParams params;
    Mat<double> U;
    Mat<double> V;
    Col<double> a;
    Col<double> b;
    Col<double> del_U;
    Col<double> del_V;

    void updateGradU(
            Col<double> *Ui,
            int y,
            Col<double> *Vj,
            double ai,
            double bj,
            double s,
            int i,
            int j
            );

    void updateGradV(
            Col<double> *Ui,
            int y,
            Col<double> *Vj,
            double ai,
            double bj,
            double s,
            int i,
            int j
            );

    double gradA(
            Col<double> *Ui,
            int y,
            Col<double> *Vj,
            double ai,
            double bj,
            double s
            );

    double gradB(
            Col<double> *Ui,
            int y,
            Col<double> *Vj,
            double ai,
            double bj,
            double s
            );

    public:
    Model(
        int M,
        int N,
        int K,
        double eta,
        double reg,
        string train_filename,
        string test_filename,
        string valid_filename,
        double mu,
        double eps = 0.01,
        double max_epochs = 40
     );
    vector<double> predict();
    double trainErr();
    double validErr();
    void train();
    void save(char *file);
    ~Model();
};
