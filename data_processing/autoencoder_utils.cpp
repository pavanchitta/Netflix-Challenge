#include <string>
#include <iostream>
#include <fstream>
#include "data.h"

using namespace std;

void fill(int *k, int size ){
	for (size_t i = 0; i < size; i++) {
		k[i] = -1;
	}
}
int* find_missing_users(string train_filename, string valid_filename, int* missing_qual) {
	int *missing_users = new int[500000];
	fill(missing_users, 500000);
	int qual_index = 0;

	int *probe_users = new int[500000];
	Data train_data = Data(train_filename);
	Data valid_data = Data(valid_filename);
	int index = 0;
	int current_user = -1;

	while (valid_data.hasNext()) {
		NetflixData d = valid_data.nextLine();
		int user = d.user;

		if (current_user != user) {
			probe_users[index] = user;
			index++;
			current_user = user;
		}
	}

	int check_index = 0;
	current_user = -1;
	int missing = 0;
	while (train_data.hasNext()) {
		NetflixData dtrain = train_data.nextLine();
		int user = dtrain.user;

		if (current_user != user) {
			if (user == probe_users[check_index]) {

				check_index++;
				if (user != current_user + 1) {
					check_index++;
				}
			}
			else {
				// if (user == missing_qual[qual_index]) {
				// 	qual_index++;
				// }

				missing_users[missing] = user;
				missing++;


			}
			current_user = user;

		}
	}

	cout << missing_users[190] << endl;
	cout << missing_users[200] << endl;
	cout << missing_users[210] << endl;



	return missing_users;
}

int* find_missing_qual(string test_filename) {
	int *missing_users = new int[500000];
	fill(missing_users, 500000);
	int missing_index = 0;
	Data test_data = Data(test_filename);
	int index = 0;
	int current_user = -1;

	while (test_data.hasNext()) {
		NetflixData d = test_data.nextLine();
		int user = d.user;

		if (current_user == -1) {
			current_user = user;
		}

		if (current_user != user) {
			if (current_user != user - 1) {
				missing_users[missing_index] = user - 1;
				missing_index++;
			}

			current_user = user;
		}
	}

	return missing_users;
}


void probe_for_pred(string valid_filename) {
	Data valid_data = Data(valid_filename);
	ofstream file;
	file.open("qual_edited.dta");

    int current_user = -1;
    int8_t *ratings = new int8_t[10000];
    int16_t *movies = new int16_t[10000];
    int size = 0;

    while (valid_data.hasNext()) {
        NetflixData d = valid_data.nextLine();
        int user = d.user;

        if (current_user == -1) {
            current_user = user;
        }

        if (current_user != user) {
            file << current_user << " ";
            for (int i = 0; i < size; i++) {
                file << (int) movies[i] << " " << (int) ratings[i];
                if (i != size - 1) {
                    file << " ";
                }
            }

            file << endl;
            size = 0;
            current_user = user;
        }

        ratings[size] = d.rating;
        movies[size] = d.movie;
        size++;
    }
	file << current_user << " ";
	   for (int i = 0; i < size; i++) {
	      file << (int) movies[i] << " " << (int) ratings[i];
	      if (i != size - 1) {
	         file << " ";
	       }
	}
	file << endl;
	file.close();
}

void train_for_pred(int* missing, string train_filename) {
	Data train_data = Data(train_filename);

    int current_user = -1;
    int8_t *ratings = new int8_t[10000];
    int16_t *movies = new int16_t[10000];
    int size = 0;
	int missing_index = 0;

	ofstream file;
	file.open("qual_4_pred_edited.dta");

    while (train_data.hasNext()) {
        NetflixData d = train_data.nextLine();
        int user = d.user;

        if (current_user == -1) {
            current_user = user;
        }

        if (current_user != user) {
			if (current_user == missing[missing_index]) {
				missing_index++;
				current_user = user;
				size = 0;
				continue;
			}





            file << current_user << " ";
            for (int i = 0; i < size; i++) {
                file << (int) movies[i] << " " << (int) ratings[i];
                if (i != size - 1) {
                    file << " ";
                }
            }

            file << endl;
            size = 0;
            current_user = user;
        }

        ratings[size] = d.rating;
        movies[size] = d.movie;
        size++;
    }
	if (current_user != missing[missing_index]) {
		file << current_user << " ";
	    for (int i = 0; i < size; i++) {
	        file << (int) movies[i] << " " << (int) ratings[i];
	        if (i != size - 1) {
	            file << " ";
	        }
	    }
	    file << endl;
	}

	file.close();
}

int main() {
    string test_filename = "/Users/matthewzeitlin/Desktop/CS156b-Netflix/data/qual.dta";
    string valid_filename = "/Users/matthewzeitlin/Desktop/CS156b-Netflix/data/probe.dta";

	int* missing_qual = find_missing_qual(test_filename);
	// int* missing = find_missing_users(test_filename, valid_filename, missing_qual);
	//cout << missing[0] << endl;
	//cout << missing[1] << endl;

	//cout << missing[2] << endl;

	train_for_pred(missing_qual, test_filename);

}
