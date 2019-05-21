#include "svd_plustime.h"
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

double Model::grad_common(int user, int rating, double b_u, double alpha_u,
                          int time, double b_i, double b_bin, double b_u_tui, double c_u, double b_f_ui,
                          Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm) {

    return (rating - GLOBAL_BIAS - dot(*Vj, *Ui + *y_norm) - b_u
            - alpha_u * this->devUser(time, this->t_u[user - 1])
            - (b_i + b_bin) * c_u - b_u_tui - b_f_ui);
}

void Model::grad_U(double del_common, Col<double> *Ui, Col<double> *Vj, int e) {
    double eta = 0.007;
    double reg = 0.015;
    this->del_U = eta * ((reg * *Ui) - (*Vj) * del_common);
}

void Model::grad_V(double del_common, Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm, int e) {
    double eta = 0.007;
    double reg = 0.015;
    this->del_V = eta * ((reg * *Vj) - (*Ui + *y_norm)* del_common);
}

double Model::grad_b_u(double del_common, double b_u) {
    double eta = 0.007;
    double reg = 0.005;
    return -eta * del_common + eta * reg * b_u;
}

double Model::grad_alpha_u(double del_common, int user, int time, double alpha_u) {
    double eta = 3.11 * pow(10, -6);
    double reg = 395 * pow(10, -2);
    //double reg = 0.015;
    return -eta * devUser(time, this->t_u[user - 1]) * del_common
           + eta * reg * alpha_u;
}

double Model::grad_b_u_tui(double del_common, double b_u_tui) {
    double eta = 0.07;
    double reg = 0.0005;
    return -eta * del_common + eta * reg * b_u_tui;
}

double Model::grad_b_i(double del_common, double b_i, double c_u) {
    double eta = 0.07;
    double reg = 0.005;
    return -eta * del_common * c_u + eta * reg * b_i;
}

double Model::grad_b_bin(double del_common, double b_bin, double c_u) {
    double eta = 0.07;
    double reg = 0.08;
    return -eta * del_common * c_u + eta * reg * b_bin;
}

double Model::grad_c_u(double del_common, double c_u, double b_i, double b_bin) {
    double eta = 5.64 * pow(10, -3);
    double reg = 4.76 * pow(10, -2);
    return -eta * del_common * (b_i + b_bin) + eta * reg * (c_u - 1);
}

double Model::grad_b_f_ui(double del_common, double b_f_ui) {
    double eta = 2.36 * pow(10, -3);
    double reg = 1.1 * pow(10, -8);
    return -eta * del_common + eta * reg * b_f_ui;
}

// Also update N(u)
void Model::user_frequency() {
    this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int time = p.date;
        int movie = p.movie;
        this->f_ui(user - 1, time) += 1;
    }
    for (int user = 0; user < this->params.M; user ++) {
        for (int time = 0; time < MAX_DATE; time ++) {
            int freq = this->f_ui(user, time);
            if (freq != 0) {
                this->f_ui(user, time) = floor(log(freq)/log(6.76));
            }
        }
    }
    this->params.Y.reset();
    cout << "Finished calculating user_frequency" << endl;
}

void Model::user_date_avg() {
    this->t_u = Col<double>(this->params.M, fill::zeros);
    Col<double> num_ratings = Col<double>(this->params.M, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int time = p.date;
        this->t_u[user - 1] += time;
        num_ratings[user - 1] += 1;
    }
    for (int i = 0; i < this->params.M; i++) {
        this->t_u[i] /= num_ratings[i];
    }
    this->params.Y.reset();
    cout << "Finished computing user_avg" << endl;
}

double Model::devUser(int time, int user_avg) {
    double beta = 0.4;
    if (time > user_avg) {
      return pow(time - user_avg, beta);
    }
    else {
      return -1 * pow(user_avg - time, beta);
    }
}

// Also update N(u)
void Model::movies_per_user() {

    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        this->N_u[user - 1].push_back(movie);
        this->N_u_size[user - 1] ++;
        //this->Ratings(movie - 1, user - 1) = rating;
    }
    cout << "Finished calculating movies_per_user" << endl;
    this->params.Y.reset();
}

double Model::trainErr() {

    int num_points = 0;
    double loss_err = 0.0;
    Col<int> seen_user = Col<int>(this->params.M, fill::zeros);

    while (this->params.Y.hasNext()) {

        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;
        int bin = time / DAYS_PER_BIN;
        int freq = this->f_ui(user - 1, time);
        if (seen_user[user - 1] == 0) {
            this->compute_y_norm(user);
            seen_user[user - 1] = 1;
        }

        loss_err += pow(rating - GLOBAL_BIAS - dot(V.col(movie - 1), U.col(user - 1) + this->Y_norm.col(user - 1))
                        - this->b_u[user - 1]
                        - (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1] -
                        this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        - this->b_u_tui(user - 1, time) - this->b_f_ui(movie - 1, freq), 2);

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
        int bin = time / DAYS_PER_BIN;
        int freq = this->f_ui(user - 1, time);

        loss_err += pow(rating - GLOBAL_BIAS - dot(V.col(movie - 1), U.col(user - 1) + this->Y_norm.col(user - 1))
                        - this->b_u[user - 1]
                        - (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1] -
                        this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        - this->b_u_tui(user - 1, time) - this->b_f_ui(movie - 1, freq), 2);

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
        int bin = time / DAYS_PER_BIN;
        int freq = this->f_ui(user - 1, time);

        Col<double> u = this->U.col(user - 1);
        Col<double> v = this->V.col(movie - 1);

        double u_m_inter = dot(v, u + this->Y_norm.col(user - 1));
        double u_bias = this->b_u[user] + this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        + this->b_u_tui(user - 1, time);
        double m_bias = (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1]
                        + this->b_f_ui(movie - 1, freq);
        double pred = GLOBAL_BIAS + u_bias + m_bias + u_m_inter;

        preds.push_back(pred);
    }
    return preds;
}

void Model::update_y_vectors(int user, Col<double>* Vj, int e) {
    vector<int> movies = this->N_u[user - 1];
    int size = this->N_u_size[user - 1];
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;
    double eta = 0.007;
    double reg = 0.015;
    Col<double> sum = Col<double>(this->params.K, fill::zeros);
    for (int movie : movies) {
        this->Y.col(movie - 1) += eta * (pow(size, -0.5) * *Vj - reg * this->Y.col(movie - 1));
        sum += this->Y.col(movie - 1);
    }
    this->Y_norm.col(user - 1) = pow(size, -0.5) * sum;

}

void Model::compute_y_norm(int user) {
    vector<int> movies = this->N_u[user -1];
    int size = this->N_u_size[user - 1];

    assert (size > 0);
    Col<double> sum = Col<double>(this->params.K, fill::zeros);
    for (int movie : movies) {
        sum += this->Y.col(movie - 1);
    }

    this->Y_norm.col(user - 1) = pow(size, -0.5) * sum;
}

void Model::train() {

    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);

    this->b_u = Col<double>(this->params.M, fill::zeros);
    this->alpha_u = Col<double>(this->params.M, fill::zeros);
    this->b_u_tui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);

    this->b_i = Col<double>(this->params.N, fill::zeros);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::zeros);
    this->c_u = Col<double>(this->params.M, fill::ones);
    this->b_f_ui = Mat<double>(this->params.N, MAX_FREQ, fill::zeros);

    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);

    this->Y = Mat<double>(this->params.K, this->params.N, fill::zeros);
    this->Y_norm = Mat<double>(this->params.K, this->params.M, fill::zeros);

    //this->Ratings = Mat<int>(this->params.N, this->params.M, fill::zeros);
    this->N_u = vector<vector<int>>(this->params.M);
    this->N_u_size = Col<int>(this->params.M, fill::zeros);

    this->movies_per_user();
    this->user_date_avg();
    //this->user_frequency();
    //this->t_u = Col<double>(this->params.M, fill::zeros);
    this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);

    this->U /= pow(10, 2);
    this->V /= pow(10, 2);

    // this->U -= 0.5 * 1/(pow(10, 2));
    // this->V -= 0.5 * 1/(pow(10, 2));


    double prev_err = validErr();
    cout << "done" << endl;
    double curr_err = 0.0;

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        Col<int> seen_user = Col<int>(this->params.M, fill::zeros);
        int count;
        Col<double> sum_v;
        Col<double> y_norm;
        while (this->params.Y.hasNext()) {

            NetflixData p = this->params.Y.nextLine();
            int user = p.user;
            int movie = p.movie;
            int rating = p.rating;
            int time = p.date;
            int bin = time / DAYS_PER_BIN;
            int freq = this->f_ui(user - 1, time);

            Col<double> u = this->U.col(user - 1);
            Col<double> v = this->V.col(movie - 1);


            // Get the SVD++ specific variables
            if (seen_user[user - 1] == 0) {
                sum_v = Col<double>(this->params.K, fill::zeros);
                count = 1;
                this->compute_y_norm(user);
                y_norm = this->Y_norm.col(user - 1);
                seen_user[user - 1] = 1;
            }

            double del_common = this->grad_common(user, rating, this->b_u[user - 1],
                    this->alpha_u[user - 1], time, this->b_i[movie - 1],
                    this->b_bin(movie - 1, bin), this->b_u_tui(user -1, time),
                    this->c_u[user - 1], this->b_f_ui(movie - 1, freq),
                    &u, &v, &y_norm);

            double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1]);
            //double del_alpha_u = this->grad_alpha_u(del_common, user, time, this->alpha_u[user - 1]);
            //double del_b_u_tui = this->grad_b_u_tui(del_common, this->b_u_tui(user - 1, time));

            double del_b_i = this->grad_b_i(del_common, this->b_i[movie - 1], this->c_u[user - 1]);
            //double del_b_bin = this->grad_b_bin(del_common, this->b_bin(movie - 1, bin), this->c_u[user - 1]);
            double del_c_u = this->grad_c_u(del_common, this->c_u[user - 1], this->b_i[movie - 1], this->b_bin(movie - 1, bin));
            //double del_b_f_ui = this->grad_b_f_ui(del_common, this->b_f_ui(movie - 1, freq));

            this->grad_U(del_common, &u, &v, e);
            this->grad_V(del_common, &u, &v, &y_norm, e);

            this->b_u[user - 1] -= del_b_u;
            //this->alpha_u[user - 1] -= del_alpha_u;
            //this->b_u_tui(user - 1, time) -= del_b_u_tui;

            this->b_i[movie - 1] -= del_b_i;
            //this->b_bin(movie - 1, bin) -= del_b_bin;
            this->c_u[user - 1] -= del_c_u;
            //this->b_f_ui(movie - 1, freq) -= del_b_f_ui;

            this->U.col(user - 1) -= this->del_U;
            this->V.col(movie - 1) -= this->del_V;
            sum_v += v * del_common;

            if (count == this->N_u_size[user - 1]) {
                update_y_vectors(user, &sum_v, e);
            }


        }

        this->params.Y.reset();
        cout << "Train Error " << trainErr() << endl;
        this->params.Y.reset();

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
