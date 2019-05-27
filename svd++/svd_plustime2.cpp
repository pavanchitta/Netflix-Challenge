#include "svd_plustime2.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <tuple>

using namespace std;

#define GLOBAL_BIAS 3.6095161972728063
#define NUM_BINS 30
#define DAYS_PER_BIN 75
#define MAX_DATE 2250
#define MAX_FREQ 5

Model::Model(
            int M,
            int N,
            int K,
            double k_eta,
            double k_reg,
            string train_filename,
            string test_filename,
            string valid_filename,
            string all_filename,
            double max_epochs

    ) : params( { M, N, K, k_eta, k_reg,Data(train_filename), Data(test_filename), Data(valid_filename), Data(all_filename), max_epochs}){

}

Model::~Model() {}

double Model::grad_common(int user, int rating, double b_u, double b_i,double b_bin,double b_u_tui,
                        double b_f_ui, double dev_alpha_u, double c_u, double c_ut,
                        Col<double> *dev_alpha_uk,
                          Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm) {

    Col<double> p_ut = *Ui + *dev_alpha_uk;

    return (rating - GLOBAL_BIAS - dot(*Vj, p_ut + *y_norm) - dev_alpha_u
            - b_u - (b_i + b_bin) * (c_u + c_ut) - b_u_tui - b_f_ui);
}

void Model::grad_U(double del_common, Col<double> *Ui, Col<double> *Vj, int e) {
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;

    double eta = params.k_eta * pow(0.9, e);
    double reg = params.k_reg;
    this->del_U = eta * ((reg * *Ui) - (*Vj) * del_common);
}

void Model::grad_V(double del_common, Col<double> *Ui, Col<double> *Vj, Col<double> *y_norm, Col<double> *dev_alpha_uk, int e) {
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;
    double eta = params.k_eta * pow(0.9, e);
    double reg = params.k_reg;
    this->del_V = eta * ((reg * *Vj) - (*Ui + *dev_alpha_uk + *y_norm)* del_common);
}

double Model::grad_b_u(double del_common, double b_u, int e) {
    double eta = 2.67 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    return -eta * del_common + eta * reg * b_u;
}

double Model::grad_b_u_tui(double del_common, double b_u_tui, int e) {
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    double eta = 2.57 * pow(10, -3);
    double reg = 0.231 * pow(10, -2);
    return -eta * del_common + eta * reg * b_u_tui;
}

double Model::grad_b_f_ui(double del_common, double b_f_ui, int e) {
    double eta = 2.36 * pow(10, -3);
    double reg = 1.1 * pow(10, -8);
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.000005;
    return -eta * del_common + eta * reg * b_f_ui;
}

double Model::grad_b_i(double del_common, double b_i, double c_u, double c_ut, int e) {
    double eta = 0.488 * pow(10, -3);
    double reg = 2.55 * pow(10, -2);
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    return -eta * del_common * (c_u + c_ut) + eta * reg * b_i;
}

double Model::grad_b_bin(double del_common, double b_bin, double c_u, double c_ut, int e) {
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    double eta = 0.115 * pow(10, -3);
    double reg = 9.29 * pow(10, -2);
    return -eta * del_common * (c_u + c_ut) + eta * reg * b_bin;
}

double Model::grad_c_u(double del_common, double c_u, double b_i, double b_bin, int e) {
    double eta = 5.64 * pow(10, -3);
    double reg = 4.76 * pow(10, -2);
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    return -eta * del_common * (b_i + b_bin) + eta * reg * (c_u - 1);
}

double Model::grad_c_ut(double del_common, double c_ut, double b_i, double b_bin, int e) {
    double eta = 1.03 * pow(10, -3);
    double reg = 1.90 * pow(10, -2);
    // double eta = 0.007;// * pow(0.9, e);
    // double reg = 0.005;
    return -eta * del_common * (b_i + b_bin) + eta * reg * (c_ut);
}

double Model::grad_alpha_u(double del_common, int user, int time, double alpha_u, int e) {
    // double eta = 0.00001;// * pow(0.9, e);
    // double reg = 10;
    double eta = 3.11 * pow(10, -6);
    double reg = 395 * pow(10, -2);
    //double reg = 0.015;
    return -eta * devUser(time, this->t_u[user - 1]) * del_common
           + eta * reg * alpha_u;
}

void Model::grad_alpha_uk(double del_common, int user, int time, Col<double>* alpha_uk, Col<double>* Vj, int e) {
    // double eta = 0.00001;// * pow(0.9, e);
    // double reg = 10;
    double eta = 1 * pow(10, -5);
    double reg = 50.0;
    //double reg = 0.015;
    this->del_alpha_uk =  -eta * devUser(time, this->t_u[user - 1]) * *Vj * del_common
           + eta * reg * *alpha_uk;
}

// void Model::grad_p_u_kt(double del_common, Col<double>* p_u_kt, Col<double> *Vj, int e) {
//     double eta = 2.57 * pow(10, -3);
//     double reg = 0.231 * pow(10, -2);
//
//     this->del_p_u_kt = eta * ((reg * *p_u_kt) - (*Vj) * del_common);
// }

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
void Model::user_frequency() {
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


// Also update N(u)
void Model::implicit_movies_per_user() {

    this->params.Y_all.reset();
    while (this->params.Y_all.hasNext()) {
        NetflixData p = this->params.Y_all.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;
        this->N_u[user - 1].push_back(movie);
        this->N_u_size[user - 1] ++;

    }
    cout << "Finished calculating movies_per_user" << endl;
    this->params.Y_all.reset();

}

// Also update N(u)
void Model::movies_per_user() {

    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;

        this->R_u[user - 1].push_back(movie);
        this->R_u_size[user - 1] ++;
        this->Rating_Time[user-1].push_back(make_tuple(rating, time));


    }
    cout << "Finished calculating movies_per_user" << endl;
    this->params.Y.reset();

}

double Model::predict_rating(int user, int movie, int time) {

    int bin = time / DAYS_PER_BIN;
    int freq = this->f_ui(user - 1, time);

    //Col<double> p_u_kt = this->p_u_kt.slice(time).col(user - 1);

    Col<double> p_ut = U.col(user - 1) + this->alpha_uk.col(user - 1)*this->devUser(time, this->t_u[user - 1])
                        + this->Y_norm.col(user - 1);
    Col<double> v = this->V.col(movie - 1);

    double pred = GLOBAL_BIAS + dot(v, p_ut) + this->b_u[user - 1] +
    (this->b_i[movie-1] + this->b_bin(movie - 1, bin)) * (this->c_u[user - 1] + this->c_ut(user - 1, time))
    + this->b_u_tui(user - 1, time) + this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
    + this->b_f_ui(movie - 1, freq);

    return pred;
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
        // int freq = this->f_ui(user - 1, time);
        int bin = time / DAYS_PER_BIN;

        if (seen_user[user - 1] == 0) {
            this->compute_y_norm(user);
            seen_user[user - 1] = 1;
        }
        // Col<double> p_ut = U.col(user - 1) + this->alpha_uk.col(user - 1)*this->devUser(time, this->t_u[user - 1])
        //                     + this->Y_norm.col(user - 1);

        // loss_err += pow(rating - GLOBAL_BIAS - dot(V.col(movie - 1), p_ut)
        //                 - this->b_u[user - 1]
        //                 - this->b_i[movie - 1]
        //                 - this->b_bin(movie - 1, bin)
        //                 - this->b_u_tui(user - 1, time)
        //                 - this->alpha_u[user - 1] * this->devUser(time, this->t_u[user - 1])
        //                 - 0, 2);

        loss_err += pow(rating - this->predict_rating(user, movie, time), 2);

        // loss_err += pow(rating - GLOBAL_BIAS - dot(V.col(movie - 1), p_ut)
        //                 - this->b_u[user - 1]
        //                 - this->b_i[movie - 1], 2);

        num_points++;
    }

    return loss_err / num_points;
}

double Model::validErr() {

    int num_points = 0;
    double loss_err = 0.0;
    //Col<int> seen_user = Col<int>(this->params.M, fill::zeros);
    this->params.Y_valid.reset();
    while (this->params.Y_valid.hasNext()) {

        NetflixData p = this->params.Y_valid.nextLine();
        int user = p.user;
        int movie = p.movie;
        int rating = p.rating;
        int time = p.date;
        //int freq = this->f_ui(user - 1, time);
        int bin = time / DAYS_PER_BIN;

        loss_err += pow(rating - this->predict_rating(user, movie, time), 2);


        num_points++;
    }
    this->params.Y_valid.reset();

    return loss_err / num_points;
}

vector<double> Model::predict() {

    vector<double> preds;

    while (this->params.Y_test.hasNext()) {

        NetflixData p = this->params.Y_test.nextLine();
        int user = p.user;
        int movie = p.movie;
        int time = p.date;
        //int freq = this->f_ui(user - 1, time);
        int bin = time / DAYS_PER_BIN;

        double pred = this->predict_rating(user, movie, time);

        preds.push_back(pred);
    }
    return preds;
}

vector<double> Model::predict_probe() {

    vector<double> preds;
    this->params.Y_valid.reset();
    while (this->params.Y_valid.hasNext()) {

        NetflixData p = this->params.Y_valid.nextLine();
        int user = p.user;
        int movie = p.movie;
        int time = p.date;
        //int freq = this->f_ui(user - 1, time);
        int bin = time / DAYS_PER_BIN;

        double pred = this->predict_rating(user, movie, time);

        preds.push_back(pred);
    }
    this->params.Y_valid.reset();
    return preds;
}

// vector<double> Model::predict_train() {
//
//     vector<double> preds;
//
//     while (this->params.Y.hasNext()) {
//
//         NetflixData p = this->params.Y.nextLine();
//         int user = p.user;
//         int movie = p.movie;
//         int time = p.date;
//         //int freq = this->f_ui(user - 1, time);
//         int bin = time / DAYS_PER_BIN;
//
//         double pred = this->predict_rating(user, movie, time);
//
//         // double pred = GLOBAL_BIAS + dot(v, p_ut) + this->b_u[user - 1] + this->b_i[movie-1];
//
//         preds.push_back(pred);
//     }
//     return preds;
// }

void Model::update_y_vectors(int user, Col<double>* Vj, int e) {
    vector<int> movies = this->N_u[user - 1];
    int size = this->N_u_size[user - 1];
    // double eta = 0.008 * pow(0.9, e);
    // double reg = 0.0015;
    double eta = params.k_eta * pow(0.9, e);
    double reg = params.k_reg;
    //double reg = 0.01;
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

void Model::writeToFileKNN(string filename, vector<double> preds) {
    ofstream out;
    out.open(filename);

    int i = 0;

    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        int time = p.date;

        out << setprecision(5) << user << " " << movie << " " << time << " " << preds[i] << "\n";
        i++;
    }
    cout << "reached" << endl;
    out.close();
}

vector<double> Model::predict_train() {
    vector<double> preds;
    this->params.Y.reset();
    while (this->params.Y.hasNext()) {
        NetflixData p = this->params.Y.nextLine();
        int user = p.user;
        int movie = p.movie;
        Col<double> u = this->U.col(user - 1);
        Col<double> v = this->V.col(movie - 1);
        double pred = this->params.mu + dot(u, v) + this->a[user - 1] + this->b[movie - 1];
        preds.push_back(pred);
    }
    return preds;
}

void Model::train() {


    this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    this->V = Mat<double>(this->params.K, this->params.N, fill::randu);

    this->b_u = Col<double>(this->params.M, fill::randu);

    this->b_i = Col<double>(this->params.N, fill::randu);
    this->b_bin = Mat<double>(this->params.N, NUM_BINS, fill::zeros);
    this->b_u_tui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->f_ui = Mat<double>(this->params.M, MAX_DATE, fill::zeros);
    this->b_f_ui = Mat<double>(this->params.N, MAX_FREQ, fill::zeros);
    this->alpha_u = Col<double>(this->params.M, fill::zeros);
    this->c_u = Col<double>(this->params.M, fill::ones);
    this->c_ut = Mat<double>(this->params.M, MAX_DATE, fill::zeros);

    this->alpha_uk = Mat<double>(this->params.K, this->params.M, fill::zeros);

    this->del_U = Col<double>(this->params.K, fill::zeros);
    this->del_V = Col<double>(this->params.K, fill::zeros);
    this->del_alpha_uk = Col<double>(this->params.K, fill::zeros);
    //this->del_p_u_kt = Col<double>(this->params.K, fill::zeros);

    //this->p_u_kt = Cube<double>(this->params.K, this->params.M, MAX_DATE, fill::zeros);

    this->Y = Mat<double>(this->params.K, this->params.N, fill::zeros);
    this->N_u = vector<vector<int>>(this->params.M);
    this->N_u_size = Col<int>(this->params.M, fill::zeros);
    this->R_u = vector<vector<int>>(this->params.M);
    this->R_u_size = Col<int>(this->params.M, fill::zeros);
    // this->R_u = vector<vector<int>>(this->params.M);
    // this->R_u_size = Col<int>(this->params.M, fill::zeros);
    this->Y_norm = Mat<double>(this->params.K, this->params.M, fill::zeros);

    // this->Ratings = Mat<int>(this->params.N, this->params.M, fill::zeros);
    // this->Times = Mat<int>(this->params.N, this->params.M, fill::zeros);

    this->Rating_Time = vector<vector<tuple<int, int>>>(this->params.M);


    this->movies_per_user();
    this->implicit_movies_per_user();
    this->user_frequency();
    this->user_date_avg();


    this->U /= pow(10, 3);
    this->V /= pow(10, 3);
    //this->Y /= 1*pow(10, 2);
    this->b_u /= 1* pow(10, 3);
    this->b_i /= 1* pow(10, 3);

    // this->U -= 0.5 * 1/(pow(10, 4));
    // this->V -= 0.5 * 1/(pow(10, 4));
    // this->Y -= 0.5 * 1/(pow(10, 4));

    //this->initialize_y_norm();


    double prev_err = validErr();
    cout << "done" << endl;
    double curr_err = 0.0;

    vector<int> users;
    for (int i = 1; i <= this->params.M; i++) users.push_back(i);

    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;
        Col<double> y_norm;

        random_shuffle(users.begin(), users.end());

        for (int user : users) {
            vector<int> movies = this->R_u[user - 1];
            vector<tuple<int, int>> rating_times = this->Rating_Time[user - 1];

            int size = this->R_u_size[user - 1];
            if (size == 0) continue;
            vector<int> indexes = vector<int>(size);
            for (int i = 0; i < size; i++)
                 indexes.push_back(i);

            std::random_shuffle(indexes.begin(), indexes.end());

            this->compute_y_norm(user);
            y_norm = this->Y_norm.col(user - 1);

            Col<double> u;
            Col<double> v;
            double del_common;
            Col<double> sum_v = Col<double>(this->params.K, fill::zeros);
            int count = 1;
            for (int i = 0; i < size; i++) {
                int movie = movies[i];
                int rating, time;
                tie(rating, time) = rating_times[i];
                if (rating == 0) continue; // skip over the qual data

                int bin = time / DAYS_PER_BIN;
                int freq = this->f_ui(user - 1, time);
                u = this->U.col(user - 1);
                v = this->V.col(movie - 1);
                Col<double> alpha_uk = this->alpha_uk.col(user - 1);
                double dev = devUser(time, this->t_u[user - 1]);
                Col<double> dev_alpha_uk = alpha_uk * dev;
                double dev_alpha_u = this->alpha_u[user - 1]* dev;
                //Col<double> p_u_kt = this->p_u_kt.slice(time).col(user - 1);

                double b_f_ui = this->b_f_ui(movie - 1, freq);
                double b_bin = this->b_bin(movie - 1, bin);
                double b_u_tui = this->b_u_tui(user - 1, time);
                double c_ut = this->c_ut(user - 1, time);

                del_common = this->grad_common(user, rating, this->b_u[user - 1], b_bin,
                            this->b_i[movie - 1],b_u_tui,b_f_ui,dev_alpha_u, this->c_u[user - 1], c_ut, &dev_alpha_uk,
                            &u, &v, &y_norm);

                double del_b_u = this->grad_b_u(del_common, this->b_u[user - 1], e);
                double del_b_i = this->grad_b_i(del_common, this->b_i[movie - 1], this->c_u[user - 1], c_ut, e);
                double del_b_bin = this->grad_b_bin(del_common, b_bin,this->c_u[user - 1], c_ut, e);
                double del_alpha_u = this->grad_alpha_u(del_common, user, time, this->alpha_u[user - 1], e);

                double del_b_u_tui = this->grad_b_u_tui(del_common, b_u_tui, e);
                double del_b_f_ui = this->grad_b_f_ui(del_common, b_f_ui, e);
                double del_c_u = this->grad_c_u(del_common, this->c_u[user - 1], this->b_i[movie - 1], b_bin, e);
                double del_c_u_t = this->grad_c_ut(del_common, c_ut, this->b_i[movie - 1], b_bin, e);

                this->grad_alpha_uk(del_common, user, time, &alpha_uk, &v, e);
                this->grad_U(del_common, &u, &v, e);
                this->grad_V(del_common, &u, &v, &y_norm, &dev_alpha_uk, e);
                //this->grad_p_u_kt(del_common, &p_u_kt, &v, e);

                this->b_u[user - 1] -= del_b_u;
                this->alpha_u[user - 1] -= del_alpha_u;
                this->b_i[movie - 1] -= del_b_i;
                this->b_bin(movie - 1, bin) -= del_b_bin;
                this->c_u[user - 1] -= del_c_u;
                this->c_ut(user - 1, time) -= del_c_u_t;
                this->U.col(user - 1) -= this->del_U;
                this->V.col(movie - 1) -= this->del_V;
                this->alpha_uk.col(user - 1) -= this->del_alpha_uk;
                //this->p_u_kt.slice(time).col(user - 1) -= this->del_p_u_kt;
                this->b_u_tui(user - 1, time) -= del_b_u_tui;
                this->b_f_ui(movie - 1, freq) -= del_b_f_ui;
                sum_v += v * del_common;


            }
            update_y_vectors(user, &sum_v, e);

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
