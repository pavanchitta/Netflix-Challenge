#include "time_svd.h"
#include <stdio.h>

using namespace std;

#define GLOBAL_BIAS 3.6095161972728063
#define NUM_BINS 33
#define DAYS_PER_BIN 70
#define MAX_DATE 2300
#define MAX_FREQ 5

TimeSVD::TimeSVD(
            int M,
            int N,
            int K,
            string train_filename,
            string test_filename,
            string valid_filename,
            double max_epochs,
            double initAvg
    ) : params( { M, N, K, Data(train_filename), Data(test_filename), Data(valid_filename), max_epochs, initAvg}){

}

TimeSVD::~TimeSVD() {}

double TimeSVD::grad_common(int user, int rating, double b_u, double alpha_u,
                          int time, double b_i, double b_bin, double b_u_tui, double c_u, double b_f_ui,
                          Col<double> *Ui, Col<double> *Vj) {
    return (rating - GLOBAL_BIAS - dot(*Ui, *Vj) - b_u
            - alpha_u * this->devUser(time, this->t_u[user - 1])
            - (b_i + b_bin) * c_u - b_u_tui - b_f_ui);
}

void TimeSVD::grad_U(double del_common, Col<double> *Ui, Col<double> *Vj, int e) {
    double eta = 0.007;// * pow(0.9, e);
    double reg = 0.01;
    this->del_U = eta * ((reg * *Ui) - (*Vj) * del_common);
}

void TimeSVD::grad_V(double del_common, Col<double> *Ui, Col<double> *Vj, int e) {
    double eta = 0.007;// * pow(0.9, e);
    double reg = 0.01;
    this->del_V = eta * ((reg * *Vj) - *Ui * del_common);
}

double TimeSVD::grad_b_u(double del_common, double b_u) {
    double eta = 2.67 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    return -eta * del_common + eta * reg * b_u;
}

double TimeSVD::grad_alpha_u(double del_common, int user, int time, double alpha_u) {
    double eta = 3.11 * pow(10, -6);
    double reg = 395 * pow(10, -2);
    //double eta = 0.01 * pow(10, -3);
    //double reg = 5000 * pow(10, -2);
    return -eta * devUser(time, this->t_u[user - 1]) * del_common
           + eta * reg * alpha_u;
}

double TimeSVD::grad_b_u_tui(double del_common, double b_u_tui) {
    double eta = 2.57 * pow(10, -3);
    double reg = 0.231 * pow(10, -2);
    return -eta * del_common + eta * reg * b_u_tui;
}

double TimeSVD::grad_b_i(double del_common, double b_i, double c_u) {
    double eta = 0.488 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    return -eta * del_common * c_u + eta * reg * b_i;
}

double TimeSVD::grad_b_bin(double del_common, double b_bin, double c_u) {
    double eta = 0.115 * pow(10, -3);
    double reg = 9.29 * pow(10, -2);
    return -eta * del_common * c_u + eta * reg * b_bin;
}

double TimeSVD::grad_c_u(double del_common, double c_u, double b_i, double b_bin) {
    double eta = 5.64 * pow(10, -3);
    double reg = 4.76 * pow(10, -2);
    return -eta * del_common * (b_i + b_bin) + eta * reg * (c_u - 1);
}

double TimeSVD::grad_b_f_ui(double del_common, double b_f_ui) {
    double eta = 2.36 * pow(10, -3);
    double reg = 1.1 * pow(10, -8);
    return -eta * del_common + eta * reg * b_f_ui;
}

void TimeSVD::user_frequency() {
    this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int time = p.date;
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
    cout << "Finished calculating user_frequency" << endl;
}

void TimeSVD::user_date_avg() {
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
    cout << "Finished computing user_avg" << endl;
}

double TimeSVD::devUser(int time, int user_avg) {
    double beta = 0.4;
    if (time > user_avg) {
      return pow(time - user_avg, beta);
    }
    else {
      return -1 * pow(user_avg - time, beta);
    }
}

double TimeSVD::trainErr() {

    int num_points = 0;
    double loss_err = 0.0;

    while (this->params.Y.hasNext()) {

        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;
        int bin = time / DAYS_PER_BIN;
        int freq = this->f_ui(user - 1, time);

        loss_err += pow(rating - GLOBAL_BIAS - dot(U.col(user - 1), V.col(movie - 1)) - this->b_u[user - 1]
                        - (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1] -
                        this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        - this->b_u_tui(user - 1, time) - this->b_f_ui(movie - 1, freq), 2);

        num_points++;
    }

    return loss_err / num_points;
}

double TimeSVD::validErr() {

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

        loss_err += pow(rating - GLOBAL_BIAS - dot(U.col(user - 1), V.col(movie - 1)) - this->b_u[user - 1]
                        - (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1] -
                        this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        - this->b_u_tui(user - 1, time) - this->b_f_ui(movie - 1, freq), 2);

        num_points++;
    }

    return loss_err / num_points;
}

vector<double> TimeSVD::predict() {

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

        double u_m_inter = dot(u, v);
        double u_bias = this->b_u[user - 1] + this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
                        + this->b_u_tui(user - 1, time);
        double m_bias = (this->b_i[movie - 1] + this->b_bin(movie - 1, bin)) * this->c_u[user - 1]
                        + this->b_f_ui(movie - 1, freq);
        double pred = GLOBAL_BIAS + u_bias + m_bias + u_m_inter;

        preds.push_back(pred);
    }
    return preds;
}

void TimeSVD::train() {

    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);

    // this->U -= 0.5;
    // this->V -= 0.5;
    this->U /= this->params.initAvg;
    this->V /= this->params.initAvg;
    this->U -= 0.5 * 1 / this->params.initAvg;;
    this->V -= 0.5 * 1 / this->params.initAvg;;

    this->b_u = Col<double>(this->params.M, fill::zeros);
    this->alpha_u = Col<double>(this->params.M, fill::zeros);
    this->b_u_tui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);

    this->b_i = Col<double>(this->params.N, fill::zeros);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::zeros);
    this->c_u = Col<double>(this->params.M, fill::ones);
    this->b_f_ui = Mat<double>(this->params.N, MAX_FREQ, fill::zeros);

    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);

    this->user_date_avg();
    this->user_frequency();
    //this->t_u = Col<double>(this->params.M, fill::zeros);
    //this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->params.Y.reset();

    double prev_err = validErr();
    cout << "done" << endl;
    double curr_err = 0.0;

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
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

            double del_common = this->grad_common(user, rating, this->b_u[user - 1],
                    this->alpha_u[user - 1], time, this->b_i[movie - 1],
                    this->b_bin(movie - 1, bin), this->b_u_tui(user -1, time),
                    this->c_u[user - 1], this->b_f_ui(movie - 1, freq),
                    &u, &v);

            double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1]);
            double del_alpha_u = this->grad_alpha_u(del_common, user, time, this->alpha_u[user - 1]);
            double del_b_u_tui = this->grad_b_u_tui(del_common, this->b_u_tui(user - 1, time));

            double del_b_i = this->grad_b_i(del_common, this->b_i[movie - 1], this->c_u[user - 1]);
            double del_b_bin = this->grad_b_bin(del_common, this->b_bin(movie - 1, bin), this->c_u[user - 1]);
            double del_c_u = this->grad_c_u(del_common, this->c_u[user - 1], this->b_i[movie - 1], this->b_bin(movie - 1, bin));
            double del_b_f_ui = this->grad_b_f_ui(del_common, this->b_f_ui(movie - 1, freq));

            this->grad_U(del_common, &u, &v, e);
            this->grad_V(del_common, &u, &v, e);

            this->b_u[user - 1] -= del_b_u;
            this->alpha_u[user - 1] -= del_alpha_u;
            this->b_u_tui(user - 1, time) -= del_b_u_tui;

            this->b_i[movie - 1] -= del_b_i;
            this->b_bin(movie - 1, bin) -= del_b_bin;
            this->c_u[user - 1] -= del_c_u;
            this->b_f_ui(movie - 1, freq) -= del_b_f_ui;

            this->U.col(user - 1) -= this->del_U;
            this->V.col(movie - 1) -= this->del_V;

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
