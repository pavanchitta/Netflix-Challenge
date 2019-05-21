#include "knn.h"
#include <stdio.h>

void KNN::loadData(string umFilename, string muFilename) {
    cout << "Loading data\n";

    string line;
    char curLine[CHARS_PER_LINE];
    int userID, movieID, time, rating;

    int i = -1, last_seen = 0, j;

    // To calculate the movie averages
    int num_ratings = 0, avg = 0;

    // use the um data to get the um pairs
    ifstream trainingDtaUm (umFilename);
    if (trainingDtaUm.fail()) {
        cout << "train.dta: Open failed (um).\n";
        exit(-1);
    }

    while (getline(trainingDtaUm, line)) {
        memcpy(curLine, line.c_str(), CHARS_PER_LINE);
        userID = atoi(strtok(curLine, " ")) - 1;         // userID is 0 indexed
        movieID = (short) atoi(strtok(NULL, " ")) - 1;
        time = atoi(strtok(NULL, " "));
        rating = (char) atoi(strtok(NULL, " "));

        if (last_seen == userID) {
            i++;
        } else {
            i = 0;
            last_seen = userID;
        }

        um[userID].push_back(um_pair());
        um[userID][i].movie = movieID;
        um[userID][i].rating = rating;
    }

    trainingDtaUm.close();
    cout << "loaded um" << endl;

    i = -1;
    last_seen = 0;

    // now use the mu data to calculate movie averages and get mu pairs
    ifstream trainingDtaMu (muFilename);
    if (trainingDtaMu.fail()) {
        cout << "train.dta: Open failed (mu).\n";
        exit(-1);
    }

    while (getline(trainingDtaMu, line)) {
        memcpy(curLine, line.c_str(), CHARS_PER_LINE);
        userID = atoi(strtok(curLine, " ")) - 1;     // userID is 0 indexed
        movieID = (short) atoi(strtok(NULL, " ")) - 1;
        time = atoi(strtok(NULL, " "));
        rating = (char) atoi(strtok(NULL, " "));

        // If we're still on the same movie
        if (last_seen == movieID) {
            i++;
            num_ratings += 1;
            avg += rating;
        } else {
            // otherwise calculate the average and move on
            i = 0;
            last_seen = movieID;
            movieAverages[movieID] = float(avg) / num_ratings;
            num_ratings = 1;
            avg = rating;
        }

        mu[movieID].push_back(mu_pair());
        mu[movieID][i].user = userID;
        mu[movieID][i].rating = rating;
    }

    trainingDtaMu.close();
    cout << "loaded mu" << endl;
}

void KNN::calculatePearsonCoefs() {
    cout << "Calculating Pearson coeffients\n";

    PearsonIntermediate temp[MOVIE_CNT];

    for (int i = 0; i < MOVIE_CNT; i++) {

        // first, zero out the intermediate Pearson value
        for (int k = 0; k < MOVIE_CNT; k++) {
            temp[k].x = 0;
            temp[k].y = 0;
            temp[k].xy = 0;
            temp[k].xx = 0;
            temp[k].yy = 0;
            temp[k].cnt = 0;
        }

        if ((i % 1000) == 0) {
            cout << ((double) i / MOVIE_CNT * 100) << " % complete\n";
        }

        // Go through each user that rated movie X, u is the index for user
        for (int u = 0; u < mu[i].size(); u++) {
            int user = mu[i][u].user;

            // Go through all the movies Y that that user rated, m is the index for movie
            for (int m = 0; m < um[user].size(); m++) {
                short movie = um[user][m].movie;

                // update the PearsonIntermediate of movie pair X and Y

                char rating_i = mu[i][u].rating;     // Rating of movie X
                char rating_j = um[user][m].rating;  // Rating of movie Y


                temp[movie].x += rating_i;       // Increment rating of movie X
                temp[movie].y += rating_j;       // Increment rating of movie Y

                temp[movie].xy += rating_i * rating_j;
                temp[movie].xx += rating_i * rating_i;
                temp[movie].yy += rating_j * rating_j;

                temp[movie].cnt += 1;  // Increment number of viewers of movies X and Y
            }
        }

        // now to actually calculate the pearson coeffients:
        // source: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        // stole some pseudocode from random site

        for (int k = 0; k < MOVIE_CNT; k++) {
            float x = temp[k].x;
            float y = temp[k].y;
            float xy = temp[k].xy;
            float xx = temp[k].xx;
            float yy = temp[k].yy;
            unsigned int cnt = temp[k].cnt;
            if (cnt == 0) {
                pearsonCoefs[i][k].p = 0;
            }
            else {
                float result = (cnt * xy - x * y) / (sqrt(cnt * xx - x*x) * sqrt(cnt * yy - y*y));
                result = (result != result) ? 0.0 : result; // check if result is NaN
                pearsonCoefs[i][k].p = result;
                pearsonCoefs[i][k].overlap = cnt;
            }
        }
    }
}

double KNN::predictRating(unsigned int movie, unsigned int user) {
    n_struct neighbors[MOVIE_CNT];
    int j = 0;
    n_struct tmp_pair;

    // source:  - https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/
    //          - https://en.wikipedia.org/wiki/Fisher_transformation
    for (unsigned int i = 0; i < um[user].size(); i++) {
        unsigned int curMovie = um[user][i].movie;

        p_struct tmp = pearsonCoefs[min(movie, curMovie)][max(movie, curMovie)];
        int common_users = tmp.overlap;

        // only consider if the current movie and movie have enough overlap
        if (common_users >= MIN_OVERLAP) {
            neighbors[j].overlap = common_users;
            neighbors[j].m_avg = movieAverages[movie];
            neighbors[j].n_avg = movieAverages[curMovie];
            neighbors[j].n_rating = um[user][i].rating;

            float pearson = tmp.p;
            neighbors[j].pearson = pearson;

            // Perform the fisher transform and its inverse to calculate p lower
            float p_l = tanh(atanh(pearson) - 1.96 / sqrt(common_users - 3));
            neighbors[j].p_l = p_l;
            neighbors[j].weight = p_l * p_l * log(common_users);
            j++;
        }
    }

    // need an extra element according to blog
    neighbors[j].overlap = 0;
    neighbors[j].m_avg = movieAverages[movie];
    neighbors[j].n_avg = 0;
    neighbors[j].n_rating = 0;
    neighbors[j].pearson = 0;
    neighbors[j].p_l = 0;
    neighbors[j].weight = log(MIN_OVERLAP);
    j++;

    // go through the neighbors array (size j) and find NUM_ELEMS of maximum weight
    priority_queue<n_struct> queue;
    for (int i = 0; i < j; i++) {
        if (queue.size() < NUM_ELEMS) {
            queue.push(neighbors[i]);
        } else if (queue.top().weight < neighbors[i].weight) {
            queue.pop();
            queue.push(neighbors[i]);
        }
    }

    int size = queue.size();
    int prediction = 0, diff = 0, denominator = 0;
    for (int i = 0; i < size; i++) {
        tmp_pair = queue.top();
        queue.pop();
        diff = tmp_pair.n_rating - tmp_pair.n_avg;
        diff = abs(diff);
        if (tmp_pair.pearson < 0) {
            diff = -diff;
        }
        prediction += tmp_pair.pearson * (tmp_pair.m_avg + diff);
        denominator += tmp_pair.pearson;
    }

    double result = ((float) prediction) / denominator;

    // validate the results:
    if (result != result) return GLOBAL_AVG;
    if (result > 5) return 5;
    if (result < 0) return 0;
    return result;
}

void KNN::generatePredictions(string qualFilename, string outputFilename) {
    cout << "Generating Predictions\n";

    string line;
    char curLine[CHARS_PER_LINE];
    int userID, movieID, time, rating;

    ifstream qual (qualFilename);
    ofstream out (outputFilename);

    if (qual.fail() || out.fail()) {
        cout << "qual.dta: Open failed.\n";
        exit(-1);
    }
    while (getline(qual, line)) {
        memcpy(curLine, line.c_str(), CHARS_PER_LINE);
        userID = atoi(strtok(curLine, " ")) - 1;
        movieID = (short) atoi(strtok(NULL, " ")) - 1;
        rating = predictRating(movieID, userID);
        out << rating << '\n';
    }

}

int main() {
    KNN *knn = new KNN();
    knn->loadData("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/train_probe.dta",
                  "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/mu/train_probe.dta");

    knn->calculatePearsonCoefs();

    knn->generatePredictions("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/qual.dta",
                             "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/knnPrediction1.dta");


}
