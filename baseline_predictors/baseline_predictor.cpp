#include "matrix_factor_bias.h"
#include <math.h>

#define GLOBAL_BIAS 3.608612994454291
#define NUM_BINS 30
#define DAYS_PER_BIN 70

Model::Model(
        int M,
        int N,
        NetflixData Y,
        double eps,
        double max_epochs
     ) {

    this->params = { M, N, Y, eps, max_epochs };
}

double Model::grad_b_u(int rating, double b_u, double alpha_u, int time,
                       double b_i, double b_bin) {

        double eta = 2.67 * pow(10, -3);
        double reg = 2.55 * pow(10, -2);

        return -2 * eta * (rating - GLOBAL_BIAS - b_u -
                           alpha_u * this->devUser(time, this->userAvg(user)) -
                           b_i - b_bin) +
                eta * reg * 2 * b_u;
}

double Model::grad_alpha_u(int rating, double b_u, double alpha_u, int time,
                       double b_i, double b_bin) {

        double eta = 3.11 * pow(10, -6);
        double reg = 395 * pow(10, -2);

        return -2 * eta * devUser(time, this-userAvg(user)) *
               (rating - GLOBAL_BIAS - b_u -
                alpha_u * this->devUser(time, this->userAvg(user)) -
                b_i - b_bin) +
                eta * reg * 2 * alpha_u;
}


double Model::grad_b_i(int rating, double b_u, double alpha_u, int time,
                       double b_i, double b_bin) {

          double eta = 0.488 * pow(10, -3);
          double reg = 2.55 * pow(10, -2);

          return -2 * eta * (rating - GLOBAL_BIAS - b_u -
                             alpha_u * this->devUser(time, this->userAvg(user)) -
                             b_i - b_bin) +
                 eta * reg * 2 * b_i;
}

double Model::grad_b_bin(int rating, double b_u, double alpha_u, int time,
                       double b_i, double b_bin) {

          double eta = 0.115 * pow(10, -3);
          double reg = 9.29 * pow(10, -2);

          return -2 * eta * (rating - GLOBAL_BIAS - b_u -
                             alpha_u * this->devUser(time, this->userAvg(user)) -
                             b_i - b_bin) +
                 eta * reg * 2 * b_bin;
}

double Model::trainErr() {
    NetflixData::iterator ptr;
    double loss_err = 0.0;

    for (ptr = this->params.Y.begin(), ptr < this->params.Y.end(); ptr++) {
        NetflixPoint p = *ptr;
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating
        int time = p.time;
        int bin = time / DAYS_PER_BIN;

        loss_err += pow(rating - GLOBAL_BIAS - this->b_u[user - 1] -
                        alpha_u[user - 1] * this->devUser(time, this->userAvg(user)) -
                        this->b_i[movie - 1] - this->b_bin[movie - 1][bin], 2);
    }

    return loss_err;
}

Model::~Model() {}

double Model::devUser(int time, int user_avg) {
    if (time > user_avg) {
      return pow(time - user_avg, beta)
    }
    else {
      return -1 * pow(user_avg - time, beta)
    }
}

void Model::train() {

    // Intialize matrices for user and movie biases.
    this->b_u = Col<double>(this->params.M, fill::randu);
    this->alpha_u = Col<double>(this->params.M, fill::randu);
    this->b_i = Col<double>(this->params.N, fill::randu);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::randu)

    // Normalize random entries to be between -0.5 and 0.5.
    this->b_u -= 0.5;
    this->alpha_u -= 0.5;
    this->b_i -= 0.5;
    this->b_bin -= 0.5;

    for (int e = 0; e < this->params.max_epochs; e++) {
        printf("Running Epoch %d............", e);
        NetflixData::iterator ptr;
        for (ptr = this->params.Y.begin(), ptr < this->params.Y.end(); ptr++) {
            NetflixPoint p = *ptr;
            int user = p.user;
            int movie = p.movie;
            int rating = p.rating;
            int time = p.time;
            int bin = time / DAYS_PER_BIN;

            double del_b_u = this->grad_b_u(rating, this->b_u[user - 1],
                    this->alpha[user - 1], time, this->b_i[movie - 1],
                    this->b_bin[movie - 1][bin]);

            double del_alpha_u = this->grad_alpha_u(rating, this->b_u[user - 1],
                    this->alpha[user - 1], time, this->b_i[movie - 1],
                    this->b_bin[movie - 1][bin]);

            double del_V = this->grad_b_bin(rating, this->b_u[user - 1],
                    this->alpha[user - 1], time, this->b_i[movie - 1],
                    this->b_bin[movie - 1][bin]);

            double del_b_i = this->grad_b_i(rating, this->b_u[user - 1],
                    this->alpha[user - 1], time, this->b_i[movie - 1],
                    this->b_bin[movie - 1][bin]);

            this->b_u[user - 1] -= del_b_u;
            this->alpha_u[user - 1] -= del_alpha_u;
            this->b_i[movie - 1] -= del_b_i;
            this->b[movie - 1][bin] -= del_b_bin;
        }

        printf("Error %f..................", trainErr());
    }
}
