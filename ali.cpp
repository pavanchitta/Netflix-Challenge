#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>


using namespace std;

class Data {
    private:
        string filename;
        vector<vector<int> > vectors;
        vector<vector<int> > fill_vector (string filename);
    public:
        Data (string f);
        vector<vector<int> >::iterator get_begin();
        vector<vector<int> >::iterator get_end();

};

Data::Data (string f) {
    vectors = fill_vector(f);
}

vector<vector<int> > Data::fill_vector (string filename) {
    ifstream csv(filename);
    vector<vector<int> > datas;
    string delimeter = " ";


    for(string line; getline(csv, line); ) {
        vector<int> data;

        auto start = 0U;
        auto end = line.find(delimeter);

        while (end != string::npos) {
            data.push_back(stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        data.push_back(stod(line.substr(start, end)));
        datas.push_back(data);
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

int main() {
    string filename = "um/sample.txt";
    Data d(filename);

    vector<vector<int> >::iterator it_begin = d.get_begin();
    vector<vector<int> >::iterator it_end = d.get_end();

    for (vector<vector<int> >::iterator it = it_begin ; it != it_end; ++it)
        print_vector(*it);

    return 0;
}
