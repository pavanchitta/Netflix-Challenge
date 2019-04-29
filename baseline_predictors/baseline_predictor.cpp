#include "baseline_predictor.h"
#include <math.h>

#define GLOBAL_BIAS 3.6095161972728063
#define NUM_BINS 33
#define DAYS_PER_BIN 70
#define MAX_DATE 2300
#define MAX_FREQ 5

Model::Model(int M, int N, string filename, double eps, double max_epochs):
params( { M, N, Data(filename), eps, max_epochs } ) {
}

Model::~Model() {}

double Model::grad_common(int user, int rating, double b_u, double alpha_u,
                          int time, double b_i, double b_bin, double b_u_tui, double c_u, double b_f_ui) {
    return (rating - GLOBAL_BIAS - b_u
            - alpha_u * this->devUser(time, this->t_u[user - 1])
            - (b_i + b_bin)*c_u - b_u_tui - b_f_ui);
}

double Model::grad_b_u(double del_common, double b_u) {
    double eta = 2.67 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    //return -2 * eta * del_common + eta * reg * 2 * b_u;
    return -1 * eta * del_common + eta * reg * 1 * b_u;
}

double Model::grad_alpha_u(double del_common, int user, int time, double alpha_u) {
    // double eta = 3.11 * pow(10, -6);
    // double reg = 395 * pow(10, -2);
    double eta = 0.01 * pow(10, -3);
    double reg = 5000 * pow(10, -2);
    // return -2 * eta * devUser(time, this->t_u[user - 1]) * del_common
    //        + eta * reg * 2 * alpha_u;
    return -1 * eta * devUser(time, this->t_u[user - 1]) * del_common
           + eta * reg * 1 * alpha_u;
}

double Model::grad_b_i(double del_common, double b_i) {
    // double eta = 0.488 * pow(10, -3);
    // double reg = 2.55 * pow(10, -2);
    double eta = 0.488 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    //return -2 * eta * del_common + eta * reg * 2 * b_i;
    return -1 * eta * del_common + eta * reg * 1 * b_i;

}

double Model::grad_b_bin(double del_common, double b_bin) {
    // double eta = 0.115 * pow(10, -6);
    // double reg = 9.29 * pow(10, -6);
    double eta = 0.05 * pow(10, -3);
    double reg = 10 * pow(10, -2);
    //return -2 * eta * del_common + eta * reg * 2 * b_bin;
    return -eta * del_common + eta * reg * b_bin;
}

double Model::grad_b_u_tui(double del_common, double b_u_tui) {
    double eta = 2.57 * pow(10, -3);
    double reg = 0.5 * pow(10, -2);
    //return -2 * eta * del_common + eta * reg * 2 * b_u_tui;
    return -1 * eta * del_common + eta * reg * 1 * b_u_tui;
}

double Model::grad_c_u(double del_common, double c_u, double b_i, double b_bin) {
    double eta = 8 * pow(10, -3);
    double reg = 1 * pow(10, -2);
    return -eta * del_common * (b_i + b_bin) + eta * reg * c_u;
}

double Model::grad_b_f_ui(double del_common, double b_f_ui) {
    double eta = 2.36 * pow(10, -3);
    double reg = 1.1 * pow(10, -8);
    return -eta * del_common + eta * reg * b_f_ui;
}


void Model::user_frequency() {
    this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int time = p.date;
        this->f_ui(user - 1, time) += 1;
    }
    this->f_ui = floor(log(this->f_ui)/log(6.76));
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

double Model::trainErr() {
    double loss_err = 0.0;
    this->params.Y.reset();
    int num_points = 0;
    while (this->params.Y.hasNext()) {
        num_points ++;
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;
        int bin = time / DAYS_PER_BIN;

        int freq = this->f_ui(user - 1, time);
        loss_err += pow(rating - GLOBAL_BIAS - this->b_u[user - 1]
                        - (this->b_i[movie - 1] + this->b_bin(movie - 1, bin))* this->c_u[user - 1] -
                        alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        - this->b_u_tui(user - 1, time) - this->b_f_ui(user - 1, freq), 2);
    }
    return loss_err/num_points;
}

void Model::train() {

    // Intialize matrices for user and movie biases.
    this->b_u = Col<double>(this->params.M, fill::randu);
    this->alpha_u = Col<double>(this->params.M, fill::randu);
    this->b_i = Col<double>(this->params.N, fill::randu);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::randu);

    this->b_u_tui = Mat<double>(this->params.M,MAX_DATE, fill::zeros);
    this->c_u = Col<double>(this->params.M, fill::zeros);
    this->b_f_ui = Mat<double>(this->params.M, MAX_FREQ, fill::randu);


    // Normalize random entries to be between -0.5 and 0.5.
    this->b_u -= 0.5;
    this->alpha_u -= 0.5;
    this->b_i -= 0.5;
    this->b_bin -= 0.5;
    this->b_f_ui -= 0.5;
    //this->c_u -= 0.5;

    this->user_date_avg();
    this->user_frequency();
    //this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);

    //this->t_u = Col<double>(this->params.M, fill::zeros);
    cout << "finished getting user_avg" << endl;
    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        this->params.Y.reset();
        cout << "reached1" << endl;
        while (this->params.Y.hasNext()) {
            NetflixData p = this->params.Y.nextLine();
            int user = p.user;
            int movie = p.movie;
            int rating = p.rating;
            int time = p.date;
            int bin = time / DAYS_PER_BIN;
            int freq = this->f_ui(user - 1, time);

            double del_common = this->grad_common(user, rating, this->b_u[user - 1],
                    this->alpha_u[user - 1], time, this->b_i[movie - 1],
                    this->b_bin(movie - 1, bin), this->b_u_tui(user -1, time), this->c_u[user - 1], this->b_f_ui(user - 1, freq));

            double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1]);
            double del_alpha_u = this->grad_alpha_u(del_common, user, time, this->alpha_u[user - 1]);
            double del_b_bin = this->grad_b_bin(del_common, this->b_bin(movie - 1, bin));
            double del_b_i = this->grad_b_i(del_common, this->b_i[movie - 1]);

            double del_b_u_tui = this->grad_b_u_tui(del_common, this->b_u_tui(user - 1, time));
            double del_c_u = this->grad_c_u(del_common, this->c_u[user - 1], this->b_i[movie - 1], this->b_bin(movie - 1, bin));
            double del_b_f_ui = this->grad_b_f_ui(del_common, this->b_f_ui(user - 1, freq));

            this->b_u[user - 1] -= del_b_u;
            this->alpha_u[user - 1] -= del_alpha_u;
            this->b_i[movie - 1] -= del_b_i;
            this->b_bin(movie - 1, bin) -= del_b_bin;
            this->b_u_tui(user - 1, time) -= del_b_u_tui;
            this->c_u[user - 1] -= del_c_u;
            this->b_f_ui(user - 1, freq) -= del_b_f_ui;

        }
        //cout << "Error " << trainErr() << endl;
        this->params.Y.reset();
    }
}
