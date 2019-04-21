#include "netflix_data.h"
#include <armadillo>

using namespace arma;

typedef struct {
    int M;
    int N;
    int K;
    double eta;
    double reg;
    NetflixData Y;
    double mu;
    double eps;
    double max_epochs;
} ModelParams;

class Model {
    ModelParams params;
    Mat<double> U;
    Mat<double> V;
    Row<double> a;
    Row<double> b;

    double gradU(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj
            );

    double gradV(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj
            );

    double gradA(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj
            );

    double gradB(
            Col<double> Ui, 
            int y, 
            Col<double> Vj, 
            double ai, 
            double bj
            );



    public:
    Model(
        int M, 
        int N, 
        int K, 
        double eta, 
        double reg, 
        NetflixData Y, 
        double mu,
        double eps = 0.01,
        double max_epochs = 200
     );

    double trainErr();
    void train();
    void save(char *file);
    ~Model();
}; 
