#ifndef NNMATRIXFACTOR_H
#define NNMATRIXFACTOR_H
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
} NNSVDParams;

class NNSVD {

    Mat<double> U;
    Mat<double> V;
    Col<double> a;
    Col<double> b;
    Col<double> t_u;
    Col<double> t_m;
    Col<double> del_U;
    Col<double> del_V;

    void movie_rating_avg();
    void user_rating_avg();

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
    NNSVDParams params;
    NNSVD(
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
        double max_epochs = 30
     );
    vector<double> predict();
    double trainErr();
    double validErr();
    void train();
    void save(char *file);
    ~NNSVD();
};
#endif
