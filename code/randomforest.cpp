#include "randomforest.h"

// #define GREEN 10
// #define YELLOW 20
// #define RED 30
// #define APPLE 40
// #define GRAPE 50
// #define LEMON 60

// make this file "build nodes file", do another .h and .cpp for random forests
data_class::data_class(){
    training_data.push_back(data_node(GREEN,3, APPLE));
    training_data.push_back(data_node(YELLOW,3,APPLE));
    training_data.push_back(data_node(RED, 1, GRAPE));
    training_data.push_back(data_node(RED, 1, GRAPE));
    training_data.push_back(data_node(YELLOW, 3, LEMON));
    //training_data.at(1)->color; // share ptr ex of derence
    for(unsigned int i = 0; i < training_data.size(); i++){
        cout<<training_data.at(i).color << " " << training_data.at(i).diameter << " "<< training_data.at(i).label << endl;    
    }
}

//using shared ptrs
data_class::~data_class(){

}

void data_class::Print_vec(vector<data_node> training_data){
    for(unsigned int i = 0; i < training_data.size(); i++){
        cout<<training_data.at(i).color << " " << training_data.at(i).diameter << " "<< training_data.at(i).label << endl;    
    }
}

vector<data_node> data_class::ret_vec(){
    return training_data;
}

 int data_class::calculate_matches(vector<data_node> training_data, string category, int datapoint){
    int num_matches = 0;
    // Pick all elements one by one 
    for (int i = 0; i < training_data.size(); i++) { 
        if ((category == "color")&& (training_data.at(i).color == datapoint)){
            num_matches++;
        }
        else if ((category == "diameter")&&(training_data.at(i).diameter == datapoint)){
            num_matches++;
        }
        else if ((category == "label")&&(training_data.at(i).label == datapoint)){
            num_matches++;
        }
    }
    return num_matches; 
  }

// 1) calculate gini impurity for each category.  
  //make function that can do this for info gain as well
float data_class::gini_Impurity(vector<data_node> training_data, string category, int datapoint){
    float num_total_events = training_data.size();
    //cout << "num_total_events : " << num_total_events << endl;
    float num_events_in_category = calculate_matches(training_data, category, datapoint);
    //cout <<  "num_events_in_category : " << num_events_in_category << endl;
    float num_not_in_cat = num_total_events - num_events_in_category;
    float prob_event = num_events_in_category/num_total_events;
    //cout << "prob event : " << prob_event << endl;
    float prob_not_event = num_not_in_cat/num_total_events;
    //cout << "prob not event : " << prob_not_event << endl;
    float gini = 1- (pow(prob_event, 2.0) + pow(prob_not_event, 2.0));
    //cout << "gini : " << gini << endl;
    return gini;
  }
 
vector<int> data_class::find_unique_vals(vector<data_node> training_data, vector<int>unique_values, string category){
    //find unique values in each category, max num will be 5
    int new_value;
    int size = training_data.size();
    for (int i = 0; i < size; i++){
        if (category == "color"){
            new_value = training_data.at(i).color;
        }
        else if (category == "diameter"){
            new_value = training_data.at(i).diameter;
        }
        else if (category == "label"){
            new_value = training_data.at(i).label;
        }
        // unique_values.push_back(new_value);
        vector<int>::iterator it = std::find (unique_values.begin(), unique_values.end(), new_value);
        //If element is found then it returns an iterator to the first element 
        //in the given range that’s equal to given element, else it returns an end of the list.
        if (it == unique_values.end()){
            unique_values.push_back(new_value);
        }
    }
    return unique_values;
}

int data_class::find_best_split(vector<data_node> training_data, string category){
    //https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php
    vector<int>unique_values;
    unique_values = find_unique_vals(training_data, unique_values, category);
    //vector<int, int> gini_impurities;
    float highest_gini = 0;
    int test_val=0;
    for(signed int i = 0; i < unique_values.size(); i++){
        float gini_impurity = gini_Impurity(training_data, category, unique_values.at(i));
        if ((highest_gini <= gini_impurity) && (unique_values.at(i)!= 0)){
            highest_gini = gini_impurity;
            test_val = unique_values.at(i);
        }
    }
    return test_val;
}

int data_class::random_num_generator(int high, int low){
    int n = high - low + 1;
    int i = rand() % n;
    if (i < 0) i = -i;
    return low + i;
}

string data_class::random_category(int high, int low){
    int num = random_num_generator(high, low);
    if (num == 1){
        return "color";
    }
    else if (num == 2){
        return "diameter";
    }
    else{
        return "label";
    }
}

vector<data_node> data_class::bootstrap(vector<data_node> training_data){
    //choose 5 random samples (with replacement) and return a vector of the samples
    vector<data_node> bootstrapped_vector;
    for(int i = 0; i < 5; i++){
        int j = random_num_generator(5, 0);  // choose one of the nodes randomly
        bootstrapped_vector.push_back(training_data[j]);
    }
    return bootstrapped_vector; 
}

shared_ptr<decision_tree_node> data_class::Load_DT(vector<data_node>training_data, string category){
    // i. initialize new dt node
    shared_ptr<decision_tree_node> root_node (new decision_tree_node);
    // ii. create bootstrapped dataset
    vector<data_node> bootstrapped_data = bootstrap(training_data);
    root_node->_data_nodes = bootstrapped_data;
    // iii. calculate information gain for category 
    int best_split = find_best_split(bootstrapped_data, category);
    root_node->best_split = best_split;
    // iv. store gini impurity of information gain/ best split 
    float gini_impurity = gini_Impurity(bootstrapped_data, category, best_split);
    root_node->gini_impurity = gini_impurity;
    // v. is_leaf is true
    root_node->isLeaf = true;
    // vi. create decision tree  shared pointer and load with training data
    return root_node;
}

shared_ptr<decision_tree_node> data_class::load_Split(shared_ptr<decision_tree_node> dt, vector<data_node> new_vec, int best_split, string category){
    vector<data_node> right_vec;
    vector<data_node> left_vec;
    if(category == "color"){
        for(int i=0; i < new_vec.size(); i++){
            if(new_vec.at(i).color == best_split){
                right_vec.push_back(new_vec.at(i));
            }
            else{
                left_vec.push_back(new_vec.at(i));
            }
        }
    }
    else if(category == "diameter"){
        for(int i=0; i < new_vec.size(); i++){
            if(new_vec.at(i).diameter == best_split){
                right_vec.push_back(new_vec.at(i));
            }
            else{
                left_vec.push_back(new_vec.at(i));
            }
        }
    }
    else{
        for(int i=0; i < new_vec.size(); i++){
            if(new_vec.at(i).label == best_split){
                right_vec.push_back(new_vec.at(i));
            }
            else{
                left_vec.push_back(new_vec.at(i));
            }
        }
    }
    shared_ptr<decision_tree_node> left_ptr (new decision_tree_node);
    shared_ptr<decision_tree_node> right_ptr(new decision_tree_node);
    if (left_vec.size() != 0){
        left_ptr->_data_nodes = left_vec;
    }
    if (right_vec.size() != 0){
        right_ptr->_data_nodes = right_vec;
    }
    dt->left = left_ptr;
    dt->right = right_ptr;
    return dt;
}

shared_ptr<decision_tree_node> data_class::Make_Random_DT(vector<data_node>training_data){
    //choose random category 
    string category1 = random_category(3, 1);
    // iv. choose second question randomly 
    // (we will only do 2 questions (a dept of 2), before sqrt(num features) is a little less than 2)
    //string category2 = random_category(3, 1);
    // make node level 0
    shared_ptr<decision_tree_node> dt = Load_DT(training_data, category1);
    // make node level 1
    //load_Split is loading the 
    //new_dt is a pointer to the node that will split its bootstrapped data into the right 
    //and left nodes, based on the best_split function
    dt = load_Split(dt, dt->_data_nodes, dt->best_split, category1);
    shared_ptr<decision_tree_node> new_dt;
    if((dt->left->_data_nodes.size())>(dt->_data_nodes.size())){
        new_dt = dt->left;
    }
    else{
        new_dt = dt->right;
    }
    string category2 = random_category(3, 1);
    new_dt = Load_DT(new_dt->_data_nodes, category2);
    return dt;
}

rf::rf(){
}

rf::~rf(){
    
}

shared_ptr<rf_btree> rf::initiate_rf_root(){
    shared_ptr<rf_btree> rf_root (new rf_btree);
    for (int i=0; i< BTREE_ORDER; i++){
        rf_root->children[i] = NULL;
    } 
    return rf_root;
}


// Finally, I could not get this function to work.... but the idea is we would insert a random decision tree as a child in the random forest for each child decloared in BTREE_ORDER
shared_ptr<rf_btree> rf::buildrandomforest(){
    data_class _DATA_CLASS;
    vector<data_node> training_data = _DATA_CLASS.ret_vec();
    rf_root = initiate_rf_root();
    for (int i = 0; i < BTREE_ORDER; i++){
        shared_ptr<decision_tree_node> rand_tree = _DATA_CLASS.Make_Random_DT(training_data);
        rf_root->children[i] = rand_tree;
    }
    return rf_root;
}
