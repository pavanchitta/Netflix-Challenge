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

    Col<double> gradU(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj,
            double s
            );

    Col<double> gradV(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj,
            double s
            );

    double gradA(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj,
            double s
            );

    double gradB(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
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
        string filename, 
        double mu,
        double eps = 0.01,
        double max_epochs = 200
     );

    double trainErr();
    void train();
    void save(char *file);
    ~Model();
}; 
