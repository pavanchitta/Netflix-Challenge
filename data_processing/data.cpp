#include <iostream>
#include <fstream>
#include <armadillo>

#include "data.h"

Data::Data (string f) {
    vectors = fill_vector(f);
}

vector<vector<int> > Data::fill_vector (string filename) {
    ifstream csv(filename);
    vector<vector<int> > datas;
    string delimeter = " ";

    int a = 0;

    for(string line; getline(csv, line); ) {
        vector<int> data;
        a++;

        auto start = 0U;
        auto end = line.find(delimeter);

        while (end != string::npos) {
            data.push_back(stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(stod(line.substr(start, end)));
        datas.push_back(data);

        if (a % 10000 == 0)
            cout << a << endl;
    }
    return datas;
}

vector<vector<int> >::iterator Data::get_begin () {
    return vectors.begin();
}

vector<vector<int> >::iterator Data::get_end () {
    return vectors.end();
}

void print_vector(vector<int> v) {
    for(auto i : v) {
        cout << i << " ";
    }
    cout << endl;
}

