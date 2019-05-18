#include "svd_plus.h"
#include <stdio.h>
#include <assert.h>

using namespace std;

#define GLOBAL_BIAS 3.6095161972728063
#define NUM_BINS 33
#define DAYS_PER_BIN 70
#define MAX_DATE 2300
#define MAX_FREQ 5

Model::Model(
            int M,
            int N,
            int K,
            string train_filename,
            string test_filename,
            string valid_filename,
            double max_epochs

    ) : params( { M, N, K, Data(train_filename), Data(test_filename), Data(valid_filename), max_epochs}){

}

Model::~Model() {}

double Model::grad_common(int user, int rating, double b_u, double b_i,
                          Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm) {

    return (rating - GLOBAL_BIAS - dot(*Ui, *Vj + *y_norm) - b_u
            - b_i);
}

void Model::grad_U(double del_common, Col<double> *Ui, Col<double> *Vj, int e) {
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;

    double eta = 0.01;
    double reg = 0.01;
    this->del_U = eta * ((reg * *Ui) - (*Vj) * del_common);
}

void Model::grad_V(double del_common, Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm, int e) {
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;
    double eta = 0.01;
    double reg = 0.01;
    this->del_V = eta * ((reg * *Vj) - (*Ui + *y_norm)* del_common);
}

double Model::grad_b_u(double del_common, double b_u) {
    // double eta = 2.67 * pow(10, -3);
    // double reg = 2.55 * pow(10, -2);
    double eta = 0.01;
    double reg = 0.01;
    return -eta * del_common + eta * reg * b_u;
}

double Model::grad_b_i(double del_common, double b_i) {
    // double eta = 0.488 * pow(10, -3);
    // double reg = 2.55 * pow(10, -2);
    double eta = 0.01;
    double reg = 0.01;
    return -eta * del_common + eta * reg * b_i;
}

// Also update N(u)
void Model::movies_per_user() {

    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int time = p.date;
        int movie = p.movie;
        this->N_u[user - 1].push_back(movie - 1);
        this->N_u_size[user - 1] ++;
    }
    cout << "Finished calculating movies_per_user" << endl;
}


double Model::trainErr() {

    int num_points = 0;
    double loss_err = 0.0;

    while (this->params.Y.hasNext()) {

        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;


        loss_err += pow(rating - GLOBAL_BIAS - dot(U.col(user - 1), V.col(movie - 1) + this->Y_norm.col(user - 1))
                        - this->b_u[user - 1]
                        - this->b_i[movie - 1], 2);

        num_points++;
    }

    return loss_err / num_points;
}

double Model::validErr() {

    int num_points = 0;
    double loss_err = 0.0;

    this->params.Y_valid.reset();
    while (this->params.Y_valid.hasNext()) {

        NetflixData p = this->params.Y_valid.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;

        // if (isnan(pow(rating - GLOBAL_BIAS - dot(U.col(user - 1), V.col(movie - 1) + this->Y_norm.col(user - 1))
        //                 - this->b_u[user - 1]
        //                 - this->b_i[movie - 1], 2))) {
        //                     cout << "Got a nan" << endl;
        //                 }
        loss_err += pow(rating - GLOBAL_BIAS - dot(U.col(user - 1), V.col(movie - 1) + this->Y_norm.col(user - 1))
                        - this->b_u[user - 1]
                        - this->b_i[movie - 1], 2);

        num_points++;
    }

    return loss_err / num_points;
}

vector<double> Model::predict() {

    vector<double> preds;

    while (this->params.Y_test.hasNext()) {

        NetflixData p = this->params.Y_test.nextLine();
        int user = p.user;
        int movie = p.movie;
        int time = p.date;

        Col<double> u = this->U.col(user - 1);
        Col<double> v = this->V.col(movie - 1);

        double pred = GLOBAL_BIAS + dot(u, v + this->Y_norm.col(user - 1)) + this->b_u[user - 1] + this->b_i[movie-1];

        preds.push_back(pred);
    }
    return preds;
}

void Model::update_y_vectors(int user, double del_common, Col<double>* Ui, int e) {
    vector<int> movies = this->N_u[user - 1];
    int size = this->N_u_size[user - 1];
    if (size == 0) {
        cout << "Got size zero" << endl;
    }
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;
    double eta = 0.01;
    double reg = 0.01;
    Col<double> sum = Col<double>(this->params.K, fill::zeros);
    for (int movie : movies) {
        this->Y.col(movie) += eta * (del_common * pow(size, -0.5) * *Ui - reg * this->Y.col(movie));
        sum += this->Y.col(movie);
    }

    if (isnan(sum[0])) {
        cout << "sum nan" << endl;
    }

    this->Y_norm.col(user - 1) = pow(size, -0.5) * sum;
}

void Model::train() {

    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);

    this->b_u = Col<double>(this->params.M, fill::randu);

    this->b_i = Col<double>(this->params.N, fill::randu);

    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);

    this->Y = Mat<double>(this->params.K, this->params.N, fill::randu);
    this->N_u = vector<vector<int>>(this->params.M);
    this->N_u_size = Col<int>(this->params.M, fill::zeros);
    this->Y_norm = Mat<double>(this->params.K, this->params.M, fill::zeros);

    this->movies_per_user();


    this->U /= pow(10, 2);
    this->V /= pow(10, 2);
    this->Y /= pow(10, 2);
    this->b_u /= pow(10, 2);
    this->b_i /= pow(10, 2);

    // this->U -= 0.5 * 1/(pow(10, 2));
    // this->V -= 0.5 * 1/(pow(10, 2));
    // this->Y -= 0.5 * 1/(pow(10, 2));


    double prev_err = validErr();
    cout << "done" << endl;
    double curr_err = 0.0;

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        Col<double> seen_user = Col<double>(this->params.M, fill::zeros);
        while (this->params.Y.hasNext()) {

            NetflixData p = this->params.Y.nextLine();
            int user = p.user;
            int movie = p.movie;
            int rating = p.rating;
            int time = p.date;

            Col<double> u = this->U.col(user - 1);
            Col<double> v = this->V.col(movie - 1);
            Col<double> y_norm = this->Y_norm.col(user - 1);
            if (isnan(y_norm[0])) {
                cout << "ynorm nan" << endl;
            }



            double del_common = this->grad_common(user, rating, this->b_u[user - 1],
                    this->b_i[movie - 1],&u, &v, &y_norm);

            if (isnan(del_common)) {
                cout << "nan" << endl;
            }

            double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1]);
            double del_b_i = this->grad_b_i(del_common, this->b_i[movie - 1]);

            this->grad_U(del_common, &u, &v, e);
            this->grad_V(del_common, &u, &v, &y_norm, e);

            this->b_u[user - 1] -= del_b_u;
            this->b_i[movie - 1] -= del_b_i;
            this->U.col(user - 1) -= this->del_U;
            this->V.col(movie - 1) -= this->del_V;

            // if (seen_user[user - 1] == 0) {
            //     update_y_vectors(user, del_common, &u, e);
            //     seen_user[user - 1] = 1;
            // }


        }

        this->params.Y.reset();
        // cout << "Train Error " << trainErr() << endl;
        // this->params.Y.reset();

        this->params.Y_valid.reset();
        curr_err = validErr();
        cout << "Probe Error " << curr_err << endl;
        this->params.Y_valid.reset();

        // Early stopping
        if (prev_err < curr_err) {
            cout << "Early stopping" << endl;
            break;
        }

        prev_err = curr_err;

    }
}
