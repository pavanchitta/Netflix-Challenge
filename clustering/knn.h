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

#define USER_CNT 458293
#define MOVIE_CNT 17770
#define RATING_CNT 98291669
#define GLOBAL_AVG 3.512599976023349
#define PROBE_RATINGS_CNT 1374739
// to read in data
#define CHARS_PER_LINE 30
// min neighbors needed to actually make pred
#define MIN_OVERLAP 16
// numnber of neighbors of maximum weight to consider
#define NUM_ELEMS 10

using namespace std;

typedef struct  {
    // from all viewers who rated both movie X and Y.

    float x;            //sum of ratings for movie X
    float y;            //sum of ratings for movie Y
    float xy;           //sum of product of ratings for movies X and Y
    float xx;           //sum of square of ratings for movie X
    float yy;           //sum of square of ratings for movie Y
    unsigned int cnt;   //number of viewers who rated both movies
} PearsonIntermediate;

typedef struct {
    unsigned int user;
    unsigned char rating;
} mu_pair;

typedef struct {
    unsigned short movie;
    unsigned char rating;
} um_pair;

typedef struct {
    float p;                // pearson coeffient
    unsigned int overlap;   // overlap count of the two movies
} p_struct;

typedef struct {
    unsigned int overlap;

    float m_avg;        // average rating for movies m and n
    float n_avg;

    float n_rating;

    float pearson;      // pearson value

    float p_l;          // pearson lower
    float weight;       // weight of the clustering
} n_struct;


int operator<(const n_struct &a, const n_struct &b) {
    return a.weight > b.weight;     // when comparing n_structs in queues, use the weights
}

class KNN {
    private:
        vector<um_pair> um[USER_CNT];                   // mu pair for all ratings
        vector<mu_pair> mu[MOVIE_CNT];                  // um pair for all ratings
        p_struct pearsonCoefs[MOVIE_CNT][MOVIE_CNT];    // pearson coeffients for all mm pairs
        float movieAverages[MOVIE_CNT];                 // avg ratings for all movies

        double predictRating(
            unsigned int movie,
            unsigned int user
        );

        double getRMSE(
            int numFeatures
        );

    public:
        void loadData(
            string umFilename,
            string muFilename
        );
        void calculatePearsonCoefs();
        void generatePredictions(string qualFilename, string outputFilename);

};
