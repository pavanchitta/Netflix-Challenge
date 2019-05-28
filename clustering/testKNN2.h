#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <queue>
#include <cmath>
#include <assert.h>

#define USER_CNT 458293
#define MOVIE_CNT 17770
#define RATING_CNTS 98291669
#define GLOBAL_AVG 3.512599976023349
#define PROBE_RATINGS_CNT 1374739
#define CHARS_PER_LINE 30

using namespace std;

struct mu_pair {
    unsigned int user;
    double rating;
};

struct um_pair {
    unsigned short movie;
    double rating;
};

struct s_inter {
    double x;            // sum of ratings of movie i
    double y;            // sum of ratings of movie j
    double xy;           // sum (rating_i * rating_j)
    double xx;           // sum (rating_i^2)
    double yy;           // sum (rating_j^2)
    unsigned int n;     // Num users who rated both movies
};


// To be stored in P
struct s_pear {
    double p;
    unsigned int common;
};


struct s_neighbors {
    unsigned int common;

    double m_avg;
    double n_avg;

    double n_rating;

    double pearson;

    double p_lower;
    double weight;
};


int operator<(const s_neighbors &a, const s_neighbors &b) {
    return a.weight > b.weight;
}



class KNN {
private:
    // um: for every user, stores (movie, rating) pairs.
    vector<um_pair> um[USER_CNT];

    // mu: for every movie, stores (user, rating) pairs.
    vector<mu_pair> mu[MOVIE_CNT];

    s_pear P[MOVIE_CNT][MOVIE_CNT];

    double predictRating(unsigned int movie, unsigned int user);
    void outputRMSE(short numFeats);
    stringstream mdata;

    double movieAvg[MOVIE_CNT];
    double movieCount[MOVIE_CNT];
    int min_overlap;
    int num_elems;
public:
    KNN(int a, int b);
    ~KNN() { };
    void loadData(string train_filename);
    void calcP();
    void output(string qual_filename, string output_filename);
};
