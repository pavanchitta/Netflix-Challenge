#include <iostream>
#include <fstream>
#include <armadillo>
#include <assert.h>

#include "data.h"

#define CHUNK 120000000
#define STREAM 0

Data::Data (string f) : csv(new ifstream(f))  {
    vec_inx = 0;
    filename = f;
    this->cache.arr = new NetflixData[CHUNK];
    this->cache.size = 0;
}

void Data::reset() {
    vec_inx = 0;
    if (STREAM) {
        csv = new ifstream(filename);
        this->cache.size = 0;
    }
}

void Data::fill_vector () {
    assert(vec_inx >= this->cache.size);

    this->cache.size = 0;

    string delimeter = " ";

    int lines_read = 0;
    for(string line; getline(*this->csv, line); lines_read++) {
        auto start = 0U;
        auto end = line.find(delimeter);

        int elem = 0;

        while (end != string::npos) {
            int data = stod(line.substr(start, end - start));

            switch (elem) {
                case 0:
                    this->cache.arr[lines_read].user = data;
                    break;
                case 1:
                    this->cache.arr[lines_read].movie = data;
                    break;
                case 2:
                    this->cache.arr[lines_read].date = data;
                    break;
                default:
                    assert(false);
            }

            start = end + delimeter.length();
            end = line.find(delimeter, start);
            elem++;
        }

        if (elem == 3) {
            this->cache.arr[lines_read].rating = stod(line.substr(start, end));

        }
        this->cache.size++;
        if (STREAM && lines_read > CHUNK) {
            break;
        }
    }

    vec_inx = 0;
}

Data::~Data() {
    free(this->cache.arr);
}

NetflixData Data::nextLine() {
    if (vec_inx >= cache.size) {
        fill_vector();
    }

    assert(cache.size > 0);

    return this->cache.arr[vec_inx++];
}

bool Data::hasNext() {
    return vec_inx < this->cache.size || !csv->eof();
}

void print_vector(vector<int> v) {
    for(auto i : v) {
        cout << i << " ";
    }
    cout << endl;
}
