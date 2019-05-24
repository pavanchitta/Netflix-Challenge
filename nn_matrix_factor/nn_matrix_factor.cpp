#include "nn_matrix_factor.h"
#include <stdio.h>
#include "math.h"

using namespace std;

NNSVD::NNSVD(
        int M,
        int N,
        int K,
        double eta,
        double reg,
        string train_filename,
        string test_filename,
        string valid_filename,
        double mu,
        double eps,
        double max_epochs


    ) : params( { M, N, K, eta, reg, Data(train_filename), Data(test_filename), Data(valid_filename), mu, eps, max_epochs}){

}


void NNSVD::updateGradU(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s,
        int i, int j) {
    //this->U.col(i - 1) -= this->params.eta * ((this->params.reg * *Ui) - (*Vj) * s);
    this->del_U = this->params.eta * ((this->params.reg * *Ui) - (*Vj) * s);
}

void NNSVD::updateGradV(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s,
        int i, int j) {
    //this->V.col(j - 1) -= this->params.eta * ((this->params.reg * *Vj) - *Ui * s);
    this->del_V = this->params.eta * ((this->params.reg * *Vj) - *Ui * s);
}

// double NNSVD::gradA(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s) {
//     return as_scalar(this->params.eta * ((this->params.reg * ai) - s));
// }
//
// double NNSVD::gradB(Col<double> *Ui, int y, Col<double> *Vj, double ai, double bj, double s) {
//     return as_scalar(this->params.eta * ((this->params.reg * bj) - s));
// }

void NNSVD::user_rating_avg() {
    this->t_u = Col<double>(this->params.M, fill::zeros);
    Col<double> num_ratings = Col<double>(this->params.M, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int rating = p.rating;
        this->t_u[user - 1] += rating;
        num_ratings[user - 1] += 1;
    }
    for (int i = 0; i < this->params.M; i++) {
        this->t_u[i] /= num_ratings[i];
    }
    cout << "Finished computing user averages" << endl;
}

void NNSVD::movie_rating_avg() {
    this->t_m = Col<double>(this->params.N, fill::zeros);
    Col<double> num_ratings = Col<double>(this->params.N, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int movie = p.movie;
        int rating = p.rating;
        this->t_m[movie - 1] += rating;
        num_ratings[movie - 1] += 1;
    }
    for (int i = 0; i < this->params.N; i++) {
        this->t_m[i] /= num_ratings[i];
    }
    cout << "Finished computing movie averages" << endl;
}

vector<double> NNSVD::predict() {
    vector<double> preds;
    while (this->params.Y_test.hasNext()) {
        NetflixData p = this->params.Y_test.nextLine();
        int user = p.user;
        int movie = p.movie;
        Col<double> u = this->U.col(user - 1);
        Col<double> v = this->V.col(movie - 1);
        double pred = dot(u, v);
        //  double pred = this->params.mu + dot(u, v) + this->a[user - 1] + this->b[movie - 1];
        preds.push_back(pred);
    }
    return preds;
}

double NNSVD::trainErr() {

    int k = 0;
    double loss_err = 0.0;
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int i = p.user;
        int j = p.movie;
        int y = p.rating;
        //cout << "Point " << a << endl;
        loss_err += pow(y - dot(U.col(i - 1), V.col(j - 1)), 2);
        // loss_err += pow((y - this->params.mu - dot(U.col(i - 1), V.col(j - 1))
        //        - a[i - 1] - b[j - 1]), 2);

        k++;
    }

    return loss_err / k;
}

double NNSVD::validErr() {
    int k = 0;
    double loss_err = 0.0;
    while (this->params.Y_valid.hasNext()) {
        NetflixData p = this->params.Y_valid.nextLine();
        int i = p.user;
        int j = p.movie;
        int y = p.rating;
        //cout << "Point " << a << endl;
        loss_err += pow(y - dot(U.col(i - 1), V.col(j - 1)), 2);
        // loss_err += pow((y - this->params.mu - dot(U.col(i - 1), V.col(j - 1))
        //         - a[i - 1] - b[j - 1]), 2);

        k++;
    }

    return loss_err / k;
}

NNSVD::~NNSVD() {}

void NNSVD::train() {
    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);
    this->a = Col<double>(this->params.M, fill::randu);
    this->b = Col<double>(this->params.N, fill::randu);
    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);

    //this->U -= 0.5;
    //this->V -= 0.5;
    this->a -= 0.5;
    this->b -= 0.5;
    this->a /= 100;
    this->b /= 100;

    this->user_rating_avg();
    this->movie_rating_avg();

    for (int i = 0; i < this->params.M; i++) {
        this->U.col(i).fill(sqrt(this->t_u[i] / this->params.K));
    }

    for (int i = 0; i < this->params.N; i++) {
        this->V.col(i).fill(sqrt(this->t_m[i] / this->params.K));
    }

    double prev_err = validErr();
    double curr_err = 0.0;
    for (int e = 0; e < this->params.max_epochs; e++) {
        if (e == 15) {
            this->params.reg = 0.01;
            Col<double> magic = this->U.col(6);
            cout << magic << endl;
        }
        cout << "Running Epoch " << e << endl;
        int count = 0;
        while (this->params.Y.hasNext()) {
            NetflixData p = this->params.Y.nextLine();
            int i = p.user;
            int j = p.movie;
            int y = p.rating;

            count++;

            Col<double> u = this->U.col(i - 1);
            Col<double> v = this->V.col(j - 1);

            // double s = as_scalar(y - this->params.mu - dot(u, v) - this->a[i - 1] - this->b[j - 1]);
            double s = as_scalar(y - dot(u, v));

            this->updateGradU(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s, i, j);
            this->updateGradV(&u, y, &v,
                    this->a[i - 1], this->b[j - 1], s, i, j);
            // double del_A = this->gradA(&u, y, &v,
            //         this->a[i - 1], this->b[j - 1], s);
            // double del_B = this->gradB(&u, y, &v,
            //         this->a[i - 1], this->b[j - 1], s);

            this->U.col(i - 1) -= this->del_U;
            this->V.col(j - 1) -= this->del_V;

            // this->a[i - 1] -= del_A;
            // this->b[j - 1] -= del_B;
        }

        cout << "Ran through points" << endl;
        this->params.Y_valid.reset();
        curr_err = validErr();
        cout << "Probe Error " << curr_err << endl;
        this->params.Y_valid.reset();
        this->params.Y.reset();

        // Early stopping
        // if (prev_err < curr_err) {
        //     break;
        // }

        prev_err = curr_err;
        cout << "Error " << trainErr() << endl;
        this->params.Y.reset();
    }
}
