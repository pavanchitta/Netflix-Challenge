#include <string>
#include <iostream>
#include "data.h"

using namespace std;

int main() {
    string train_filename = "/Users/vigneshv/code/CS156b-Netflix/data/train2.dta";
    string test_filename = "/Users/vigneshv/code/CS156b-Netflix/data/qual.dta";
    string valid_filename = "/Users/vigneshv/code/CS156b-Netflix/data/probe.dta";

    Data train_data = Data(train_filename);
    
    int current_user = -1;
    int8_t *ratings = new int8_t[10000];
    int16_t *movies = new int16_t[10000];
    int size = 0;

    while (train_data.hasNext()) {
        NetflixData d = train_data.nextLine();
        int user = d.user;

        if (current_user == -1) {
            current_user = user;
        }
        
        if (current_user != user) {
            cout << current_user << " ";
            for (int i = 0; i < size; i++) {
                cout << (int) movies[i] << " " << (int) ratings[i];
                if (i != size - 1) {
                    cout << " ";
                }
            }

            cout << endl;
            size = 0;
            current_user = user;
        }

        ratings[size] = d.rating;
        movies[size] = d.movie;
        size++;
    }

    cout << current_user << " ";
    for (int i = 0; i < size; i++) {
        cout << (int) movies[i] << " " << (int) ratings[i];
        if (i != size - 1) {
            cout << " ";
        }
    }

    cout << endl;

}
