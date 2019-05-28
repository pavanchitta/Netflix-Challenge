#include "testKNN2.h"

KNN::KNN(int a, int b)
{
    this->min_overlap = a;
    this->num_elems = b;
}

void KNN::loadData(string train_filename) {
    string line;
    char c_line[CHARS_PER_LINE];
    int userId;
    int movieId;
    int time;
    double rating;

    int j;

    int i = -1;
    int last_seen = 0;

    // Used for movie avgs
    int RATING_CNTs = 0;
    int avg = 0;

    ifstream trainingDta (train_filename);
    if (trainingDta.fail()) {
        cout << "train.dta: Open failed.\n";
        exit(-1);
    }
    while (getline(trainingDta, line)) {
        memcpy(c_line, line.c_str(), CHARS_PER_LINE);
        userId = atoi(strtok(c_line, " ")) - 1; // sub 1 for zero indexed
        movieId = (short) atoi(strtok(NULL, " ")) - 1;
        time = atoi(strtok(NULL, " "));
        rating = atof(strtok(NULL, " "));

        um[userId].push_back(um_pair());
        um[userId][um[userId].size() - 1].movie = movieId;
        um[userId][um[userId].size() - 1].rating = rating;
    }
    trainingDta.close();

    cout << "Loaded um" << endl;

    i = -1;
    last_seen = 0;

    // Repeat again, now for mu dta
    ifstream trainingDtaMu (train_filename);
    if (trainingDtaMu.fail()) {
        cout << "train-mu.dta: Open failed.\n";
        exit(-1);
    }


    while (getline(trainingDtaMu, line)) {
        memcpy(c_line, line.c_str(), CHARS_PER_LINE);
        userId = atoi(strtok(c_line, " ")) - 1; // sub 1 for zero indexed
        movieId = (short) atoi(strtok(NULL, " ")) - 1;
        time = atoi(strtok(NULL, " "));
        rating = atof(strtok(NULL, " "));

        movieAvg[movieId]+=rating;
        movieCount[movieId]++;

        mu[movieId].push_back(mu_pair());
        mu[movieId][mu[movieId].size() - 1].user = userId;
        mu[movieId][mu[movieId].size() - 1].rating = rating;
    }

    for (int i = 0; i < MOVIE_CNT; i++) {
        movieAvg[i] /= movieCount[i];
    }
    trainingDtaMu.close();
    cout << "Loaded mu" << endl;

}


void KNN::calcP() {
    int i, j, u, m, user, z;
    double rmse, rmse_last;
    short movie;
    double x, y, xy, xx, yy;
    unsigned int n;

    double rating_i, rating_j;

    // Vector size
    int size1, size2;

    // Intermediates for every movie pair
    s_inter tmp[MOVIE_CNT];

    cout << "Calculating P" << endl;

    rmse_last = 0;
    rmse = 2.0;

    double tmp_f;


    // Compute intermediates
    for (i = 0; i < MOVIE_CNT; i++) {

        // Zero out intermediates
        for (z = 0; z < MOVIE_CNT; z++) {
            tmp[z].x = 0;
            tmp[z].y = 0;
            tmp[z].xy = 0;
            tmp[z].xx = 0;
            tmp[z].yy = 0;
            tmp[z].n = 0;
        }

        size1 = mu[i].size();

        if ((i % 1000) == 0) {
            cout << ((double) i / MOVIE_CNT * 100) << " % complete\n";
        }

        // For each user that rated movie i
        for (u = 0; u < size1; u++) {
            user = mu[i][u].user;

            size2 = um[user].size();
            // For each movie j rated by current user
            for (m = 0; m < size2; m++) {
                movie = um[user][m].movie; // id of movie j

                // Rating of movie i
                rating_i = mu[i][u].rating;

                // Rating of movie j
                rating_j = um[user][m].rating;

                // Increment rating of movie i
                tmp[movie].x += rating_i;

                // Increment rating of movie j
                tmp[movie].y += rating_j;

                tmp[movie].xy += rating_i * rating_j;
                tmp[movie].xx += rating_i * rating_i;
                tmp[movie].yy += rating_j * rating_j;

                // Increment number of viewers of movies i AND j
                tmp[movie].n += 1;
            }
        }

        // Calculate Pearson coeff. based on:
        // https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        for (z = 0; z < MOVIE_CNT; z++) {
            x = tmp[z].x;
            y = tmp[z].y;
            xy = tmp[z].xy;
            xx = tmp[z].xx;
            yy = tmp[z].yy;
            n = tmp[z].n;
            if (n == 0 || n == 1) {
                P[i][z].p = 0;
            }
            else {
                tmp_f = (n * xy - x * y) / (sqrt(n * xx - x*x) * sqrt(n * yy - y*y));

                // if NaN
                if (tmp_f != tmp_f) {
                    // cout << n << endl;
                    // cout << "NAN" << endl;
                    tmp_f = 0.0;
                }
                // if (!(tmp_f >= -1 && tmp_f <= 1)) {
                //     cout << tmp_f << endl;
                //     cout << "Movie i " << i << " Movie z " << z << endl;
                // }
                // else {
                //     cout << "Good value " << tmp_f << endl;
                // }
                //cout << tmp_f << endl;
                //assert (tmp_f >= -1.0 && tmp_f <= 1.0);
                P[i][z].p = tmp_f;
                P[i][z].common = n;
            }
        }

    }

    cout << "P calculated" << endl;

}

double KNN::predictRating(unsigned int movie, unsigned int user) {

    double prediction = 0;
    double denom = 0;
    double diff;
    double result;

    unsigned int size, i, n;

    s_pear tmp;

    s_neighbors neighbors[MOVIE_CNT];

    priority_queue<s_neighbors> q;

    s_neighbors tmp_pair;

    double p_lower, pearson;

    int common_users;

    // Len neighbors
    int j = 0;

    // For each movie rated by user
    size = um[user].size();

    for (i = 0; i < size; i++) {
        n = um[user][i].movie; // n: movie watched by user

        tmp = P[min(movie, n)][max(movie, n)];
        common_users = tmp.common;

        // If movie and m2 have >= min_overlap viewers
        if (common_users >= min_overlap) {
            neighbors[j].common = common_users;
            neighbors[j].m_avg = movieAvg[movie];
            neighbors[j].n_avg = movieAvg[n];

            neighbors[j].n_rating = um[user][i].rating;

            pearson = tmp.p;
            neighbors[j].pearson = pearson;

            // Fisher and inverse-fisher transform (from wikipedia)
            p_lower = tanh(atanh(pearson) - 1.96 / sqrt(common_users - 3));
//             p_lower = pearson;
            neighbors[j].p_lower = p_lower;
            neighbors[j].weight = p_lower * p_lower * log(common_users);
            j++;
        }

    }

    // Add the dummy element described in the blog
    neighbors[j].common = 0;
    neighbors[j].m_avg = movieAvg[movie];
    neighbors[j].n_avg = 0;

    neighbors[j].n_rating = 0;

    neighbors[j].pearson = 0;

    neighbors[j].p_lower = 0;
    neighbors[j].weight = log(min_overlap);
    j++;

    // For each movie-pair in neighbors
    for (i = 0; i < j; i++) {
        // If there is place in queue, just push it
        if (q.size() < num_elems) {
            q.push(neighbors[i]);
        }

        // Else, push it only if this pair has a higher weight than the top
        // (smallest in top-num_elems).
        // Remove the current top first
        else {
            if (q.top().weight < neighbors[i].weight) {
                q.pop();
                q.push(neighbors[i]);
            }
        }
    }

    // Now we can go ahead and calculate rating
    size = q.size();
    //cout << "size " << size << endl;
    for (i = 0; i < size; i++) {
        tmp_pair = q.top();
        q.pop();
        diff = tmp_pair.n_rating - tmp_pair.n_avg;
        //cout << "diff " << diff << endl;
        if (tmp_pair.pearson < 0) {
            diff = -diff;
        }
        //cout << "pearson " << tmp_pair.pearson << " m_avg " << tmp_pair.m_avg << endl;
        prediction += tmp_pair.weight * (tmp_pair.m_avg + diff);
        //cout << "pred" << prediction << endl;
        denom += tmp_pair.weight;

    }



    result = ((double) prediction) / denom;

    // If result is nan, return avg
    if (result != result) {
        return tmp_pair.m_avg;
    }

    if (result > 5) {
        cout << "prediction " << prediction << endl;
        cout << "n_rating" << tmp_pair.n_rating << endl;
        cout << "m_avg " << tmp_pair.m_avg << endl;
        cout << "Result " << result << endl;
        cout << "denom " << denom << endl;
        cout << endl;
        exit(1);
    }
    /*else if (result < 1) {
        return 1;
    }
    else if (result > 5) {
        return 5;
    }*/

    return result;

}

void KNN::output(string qual_filename, string output_filename) {
    string line;
    char c_line[CHARS_PER_LINE];
    int userId;
    int movieId;
    double rating;
    stringstream fname;

    cout << "Generating output" << endl;

    fname << "../results/output" << mdata.str();

    ifstream qual (qual_filename);
    ofstream out (output_filename);
    if (qual.fail() || out.fail()) {
        cout << "qual.dta: Open failed.\n";
        exit(-1);
    }
    while (getline(qual, line)) {
        memcpy(c_line, line.c_str(), CHARS_PER_LINE);
        userId = atoi(strtok(c_line, " ")) - 1;
        movieId = (short) atoi(strtok(NULL, " ")) - 1;
        rating = predictRating(movieId, userId);
        out << rating << '\n';
    }

    cout << "Output generated" << endl;
}

int main() {
    KNN *knn = new KNN(10, 50);
    knn->loadData("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/viggy_residuls.dta");
    knn->calcP();
    knn->output("/Users/aliboubezari/Desktop/CS/CS156b-Netflix/data/um/qual.dta",
                "/Users/aliboubezari/Desktop/CS/CS156b-Netflix/outputs/viggy_model.dta");

    cout << "KNN completed.\n";

    return 0;
}
