#ifndef MATRIXFACTORBIAS_H
#define MATRIXFACTORBIAS_H
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
} SVDParams;

class SVD {

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
    SVDParams params;
    SVD(
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
        double max_epochs = 1
     );
    vector<double> predict();
    vector<double> predict_probe();
    vector<double> predict_train();
    void writeToFileKNN(string filename, vector<double> preds);
    double trainErr();
    double validErr();
    void train();
    void save(char *file);
    ~SVD();
};
#endif
