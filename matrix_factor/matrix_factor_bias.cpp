#include "matrix_factor_bias.h"

Model::Model(
        int M, 
        int N, 
        int K, 
        double eta, 
        double reg, 
        NetflixData Y, 
        double mu,
        double eps,
        double max_epochs
     ) {
    
    this->params = { M, N, K, eta, reg, Y, mu, eps, max_epochs };
}

double Model::gradU(Col<double> Ui, int y, Col<double> Vj, double ai, double bj) {
    return as_scalar(this->params.eta * ((this->params.reg * Ui) - Vj * 
            (y - this->params.mu - dot(Ui, Vj) - ai - bj)));
}

double Model::gradV(Col<double> Ui, int y, Col<double> Vj, double ai, double bj) {
    return as_scalar(this->params.eta * ((this->params.reg * Vj) - Ui * 
           (y - this->params.mu - dot(Ui, Vj) - ai - bj)));
} 

double Model::gradA(Col<double> Ui, int y, Col<double> Vj, double ai, double bj) {
    return as_scalar(this->params.eta * ((this->params.reg * ai) - 
          (y - this->params.mu - dot(Ui, Vj) - ai - bj)));
}

double Model::gradB(Col<double> Ui, int y, Col<double> Vj, double ai, double bj) {
    return as_scalar(this->params.eta * ((this->params.reg * bj) - 
          (y - this->params.mu - dot(Ui, Vj) - ai - bj)));
}

double Model::trainErr() {
    NetflixData::iterator ptr;
    double loss_err = 0.0;

    for (ptr = this->params.Y.begin(), ptr < this->params.Y.end(); ptr++) {
        NetflixPoint p = *ptr;
        int i = p.i;
        int j = p.j;
        int y = p.y;

        loss_err += pow((y - this->params.mu - dot(U.col(i - 1), V.col(j - 1))
                - a[i - 1] - b[j - 1]), 2);
    }

    return loss_err;
}

Model::~Model() {}

void Model::train() {
    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);
    this->a = Col<double>(this->params.M, fill::randu);
    this->b = Col<double>(this->params.N, fill::randu);


    this->U -= 1;
    this->V -= 1;
    this->a -= 1;
    this->b -= 1;

    for (int e = 0; e < this->params.max_epochs; e++) {
        printf("Running Epoch %d............", e);
        NetflixData::iterator ptr;
        for (ptr = this->params.Y.begin(), ptr < this->params.Y.end(); ptr++) {
            NetflixPoint p = *ptr;
            int i = p.i;
            int j = p.j;
            int y = p.y;

            double del_U = this->gradU(this->U.col(i - 1), p.y, this->V.col(j - 1), 
                    this->a[i - 1], this->b[i - 1]);
            double del_V = this->gradV(this->U.col(i - 1), p.y, this->V.col(j - 1), 
                    this->a[i - 1], this->b[i - 1]);
            double del_A = this->gradA(this->U.col(i - 1), p.y, this->V.col(j - 1), 
                    this->a[i - 1], this->b[i - 1]);
            double del_B = this->gradB(this->U.col(i - 1), p.y, this->V.col(j - 1), 
                    this->a[i - 1], this->b[i - 1]);

            
            this->U.col(i - 1) -= del_U;
            this->V.col(j - 1) -= del_V;
            this->a[i - 1] -= del_A;
            this->b[j - 1] -= del_B;
        }

        printf("Error %f..................", trainErr());
    }
}
