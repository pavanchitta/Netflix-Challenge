#include "baseline_predictor.h"
#include <math.h>

#define GLOBAL_BIAS 3.608612994454291
#define NUM_BINS 40
#define DAYS_PER_BIN 70

Model::Model(int M, int N, string filename, double eps, double max_epochs):
params( { M, N, Data(filename), eps, max_epochs } ) {
}

Model::~Model() {}

double Model::grad_common(int user, int rating, double b_u, double alpha_u,
                          int time, double b_i, double b_bin) {
    //cout << "reached" << endl;
    return (rating - GLOBAL_BIAS - b_u
            - alpha_u * this->devUser(time, this->t_u[user - 1])
            - b_i - b_bin);
}

double Model::grad_b_u(double del_common, double b_u) {
    double eta = 2.67 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    return -2 * eta * del_common + eta * reg * 2 * b_u;
}

double Model::grad_alpha_u(double del_common, int user, int time, double alpha_u) {
    double eta = 3.11 * pow(10, -6);
    double reg = 395 * pow(10, -2);
    return -2 * eta * devUser(time, this->t_u[user - 1]) * del_common
           + eta * reg * 2 * alpha_u;
}

double Model::grad_b_i(double del_common, double b_i) {
    double eta = 0.488 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    return -2 * eta * del_common + eta * reg * 2 * b_i;
}

double Model::grad_b_bin(double del_common, double b_bin) {
    double eta = 0.115 * pow(10, -3);
    double reg = 9.29 * pow(10, -2);
    return -2 * eta * del_common + eta * reg * 2 * b_bin;
}

void Model::user_date_avg(Data Y) {
    this->t_u = Col<double>(this->params.M, fill::zeros);
    Col<double> num_ratings = Col<double>(this->params.M, fill::zeros);
    this->params.Y.reset();
    int i = 0;
    while (this->params.Y.hasNext()) {
        //cout <<  i << endl;
        i++;
        vector<int> p = this->params.Y.nextLine();
        int user = p[0];
        int time = p[2];
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
        vector<int> p = this->params.Y.nextLine();
        int user = p[0];
        int movie = p[1];
        int rating = p[3];
        int time = p[2];
        int bin = time / DAYS_PER_BIN;
        loss_err += pow(rating - GLOBAL_BIAS - this->b_u[user - 1] -
                        alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1]) -
                        this->b_i[movie - 1] - this->b_bin(movie - 1, bin), 2);
    }
    return loss_err/num_points;
}

void Model::train() {

    // Intialize matrices for user and movie biases.
    this->b_u = Col<double>(this->params.M, fill::randu);
    this->alpha_u = Col<double>(this->params.M, fill::randu);
    this->b_i = Col<double>(this->params.N, fill::randu);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::randu);
    // Normalize random entries to be between -0.5 and 0.5.
    this->b_u -= 0.5;
    this->alpha_u -= 0.5;
    this->b_i -= 0.5;
    this->b_bin -= 0.5;
    // Initialize the mean date of user ratings
    //cout << "Calculating user date avgs" << endl;

    //this->t_u = Col<double>(this->params.M, fill::randu);
    this->user_date_avg(this->params.Y);

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        this->params.Y.reset();
        while (this->params.Y.hasNext()) {

            vector<int> p = this->params.Y.nextLine();
            int user = p[0];
            int movie = p[1];
            int rating = p[3];
            int time = p[2];
            int bin = time / DAYS_PER_BIN;
            //cout << time << endl;
            //cout << "reached" << endl;
            double del_common = this->grad_common(user, rating, this->b_u[user - 1],
                    this->alpha_u[user - 1], time, this->b_i[movie - 1],
                    this->b_bin(movie - 1, bin));
            //cout << "reached" << endl;
            double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1]);
            double del_alpha_u = this->grad_alpha_u(del_common, user, time, this->alpha_u[user - 1]);
            double del_b_bin = this->grad_b_bin(del_common, this->b_i[movie - 1]);
            double del_b_i = this->grad_b_i(del_common, this->b_bin(movie - 1, bin));

            this->b_u[user - 1] -= del_b_u;
            this->alpha_u[user - 1] -= del_alpha_u;
            this->b_i[movie - 1] -= del_b_i;
            this->b_bin(movie - 1, bin) -= del_b_bin;

        }

        cout << "Error " << trainErr() << endl;
    }
}