#include "matrix_factor_bias.h"
#include <stdio.h>

using namespace std;

Model::Model(
        int M,
        int N,
        int K,
        double eta,
        double reg,
        string train_filename,
        string test_filename,
        double mu,
        double eps,
        double max_epochs


    ) : params( { M, N, K, eta, reg, Data(train_filename), Data(test_filename), mu, eps, max_epochs}){

}

void Model::updateGradU(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s,
        int i, int j) {
    //this->U.col(i - 1) -= this->params.eta * ((this->params.reg * *Ui) - (*Vj) * s);
    this->del_U = this->params.eta * ((this->params.reg * *Ui) - (*Vj) * s);
}

void Model::updateGradV(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s,
        int i, int j) {
    //this->V.col(j - 1) -= this->params.eta * ((this->params.reg * *Vj) - *Ui * s);
    this->del_V = this->params.eta * ((this->params.reg * *Vj) - *Ui * s);
}

double Model::gradA(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s) {
    return as_scalar(this->params.eta * ((this->params.reg * ai) - s));
}

double Model::gradB(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s) {
    return as_scalar(this->params.eta * ((this->params.reg * bj) - s));
}

vector<double> Model::predict() {
    vector<double> preds;
    while (this->params.Y_test.hasNext()) {
        vector<int> p = this->params.Y_test.nextLine();
        int user = p[0];
        int movie = p[1];
        Col<double> u = this->U.col(user - 1);
        Col<double> v = this->V.col(movie - 1);
        double pred = this->params.mu + dot(u, v) + this->a[user - 1] + this->b[movie - 1];
        preds.push_back(pred);
    }
    return preds;
}
double Model::trainErr() {
    vector<vector<int> >::iterator ptr;

    int k = 0;
    double loss_err = 0.0;
    while (this->params.Y.hasNext()) {
        vector<int> p = this->params.Y.nextLine();
        int i = p[0];
        int j = p[1];
        int y = p[3];
        //cout << "Point " << a << endl;
        loss_err += pow((y - this->params.mu - dot(U.col(i - 1), V.col(j - 1))
                - a[i - 1] - b[j - 1]), 2);

        k++;
    }

    return loss_err / k;
}

Model::~Model() {}

void Model::train() {
    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);
    this->a = Col<double>(this->params.M, fill::randu);
    this->b = Col<double>(this->params.N, fill::randu);
    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);

    this->U -= 0.5;
    this->V -= 0.5;
    this->a -= 0.5;
    this->b -= 0.5;

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        int count = 0;
        while (this->params.Y.hasNext()) {
            vector<int> p = this->params.Y.nextLine();
            int i = p[0];
            int j = p[1];
            int y = p[3];
            //cout << "Point " << a << endl;
            count++;

            Col<double> u = this->U.col(i - 1);
            Col<double> v = this->V.col(j - 1);

            double s = as_scalar(y - this->params.mu - dot(u, v) - this->a[i - 1] - this->b[j - 1]);


            this->updateGradU(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s, i, j);
            this->updateGradV(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s, i, j);
            double del_A = this->gradA(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s);
            double del_B = this->gradB(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s);
            this->U.col(i - 1) -= this->del_U;
            this->V.col(j - 1) -= this->del_V;

            this->a[i - 1] -= del_A;
            this->b[j - 1] -= del_B;
        }

        this->params.Y.reset();
        cout << "Error " << trainErr() << endl;
        this->params.Y.reset();
    }
}
