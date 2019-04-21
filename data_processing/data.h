#include <string>
#include <vector>

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
