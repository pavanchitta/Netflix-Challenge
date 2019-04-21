#include <iostream>
#include <fstream>
#include <armadillo>
#include <assert.h>

#include "data.h"

#define CHUNK 10000

Data::Data (string f) : csv(new ifstream(f))  {
    vec_inx = 0;
    filename = f;
}

void Data::reset() {
    vec_inx = 0;
    csv = new ifstream(filename);
}

void Data::fill_vector () {
    assert(vec_inx >= this->vectors.size());

    this->vectors.clear();

    string delimeter = " ";

    int lines_read = 0;

    for(string line; getline(*this->csv, line); lines_read++) {
        if (lines_read > CHUNK) {
            break;
        }

        lines_read++;

        vector<int> data;

        auto start = 0U;
        auto end = line.find(delimeter);

        while (end != string::npos) {
            data.push_back(stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(stod(line.substr(start, end)));
        this->vectors.push_back(data);
    }

    vec_inx = 0;
}

vector<int> Data::nextLine() {
    if (vec_inx >= this->vectors.size()) {
        fill_vector();
    }

    assert(this->vectors.size() > 0);
    
    return this->vectors[vec_inx++];
}

bool Data::hasNext() {
    return vec_inx < this->vectors.size() || !csv->eof(); 
}

void print_vector(vector<int> v) {
    for(auto i : v) {
        cout << i << " ";
    }
    cout << endl;
}

