#ifndef RANDOMFOREST_H__             
#define RANDOMFOREST_H__

#include <iostream>
#include <string>
#include <fstream>
#include <iomanip> //used for formatting outputs
#include <sstream>
#include <vector>
#include <fstream>
#include <memory>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;

#define GREEN 10
#define YELLOW 20
#define RED 30
#define APPLE 40
#define GRAPE 50
#define LEMON 60
#define BTREE_ORDER 100

// The data_node class creates a dynamic array that holds the training data
class data_node{
  public:
  int color;
  int diameter;
  int label;  
  data_node(int c, int d, int l){
    color = c;
    diameter = d;
    label = l;
  }
};

// The decision_tree_node struct creates nodes for each level of the decision tree.  The node holds:
// - best_split, which is the best “question” to based on information gain
// -vector<data_node> _data_nodes  holds the bootstrapped dataset
// - float gini_impurity is the calculated mini impurity based on a randomly selected category
// -bool isLeaf tracks if the level of the node- in this case, we only will have a decision tree with a depth of 2
// - left and right pointers point to the second level of the dt or to null
struct decision_tree_node{
  int best_split;
  vector<data_node> _data_nodes;
  float gini_impurity;
  bool isLeaf;
  shared_ptr<decision_tree_node>left; 
  shared_ptr<decision_tree_node>right;

};

class data_class{
public:
// The data class constructor loads the original training dataset.  This is a very small dataset- only 5 samples- that I used to model this project.  
// If I were to build an actual random forest I would use a much larger dataset.
  data_class();

// Using shared_ptrs, no need to call deconstructor 
  ~data_class();
// Training_data is a publically available structure so it can be called outside the class
  vector<data_node> training_data;

// Testing function for accessing vector.
  vector<data_node> ret_vec();
// Testing function for accessing vector. 
  void Print_vec(vector<data_node> training_data);

// Calculate_matches finds the amount of items in each category.  This is necessary for calculating the ginning impurity.
  int calculate_matches(vector<data_node> training_data, string category, int datapoint);

// Calculated gini impurity to determine the best "split" or "question"- for example, out of the colors, which has the lowest impurity?
// if green has the lowest impurity, then we base the split on whether the color == green
// gini impurity formulas and description: https://towardsdatascience.com/gini-index-vs-information-entropy-7a7e4fed3fcb
  float gini_Impurity(vector<data_node> training_data, string category, int datapoint);

// Helper function for finding the best split- finds the variables in each category.  This changes based on bootstrapping.  
// For example, some bootstrapped datasets would have red, yellow, and green fruit, some only red, some red, and yellow, etc.
  vector<int> find_unique_vals(vector<data_node> training_data, vector<int>unique_values, string category);

// The best split it the variable in the category (columns in dataset) that has the lowest gini impurity 
  int find_best_split(vector<data_node> training_data, string category);

// Helper function to randomize bagging/ bootstrapping
  int random_num_generator(int high, int low);

// Helper function to randomize bagging/ bootstrapping
  string random_category(int high, int low);

 // Choose 5 random samples (with replacement) and return a vector of the samples
  vector<data_node> bootstrap(vector<data_node> training_data);

// i. Initialize new dt node
// ii. Create bootstrapped dataset
// iii. Calculate information gain for category 
// iv. Store gini impurity of information gain/ best split 
// v. is_leaf is true
// vi. Create decision tree  shared pointer and load with training data
  shared_ptr<decision_tree_node> Load_DT(vector<data_node>training_data, string category);

// After determining the best split, you need to sort the variables into the (false) and right (tree) child nodes.
  shared_ptr<decision_tree_node> load_Split(shared_ptr<decision_tree_node> dt, vector<data_node> new_vec, int best_split, string category);

// The culminating function of the data_class.
// We randomize the decision trees so we can compensate for bias in training data and prevent underfitting.
// I based some of my design off of this tutorial (including the dataset), but wrote all my code from scratch: https://www.youtube.com/watch?v=LDRbO9a6XPU
  shared_ptr<decision_tree_node> Make_Random_DT(vector<data_node>training_data);

private:

};

struct rf_btree{
  // Children is an array of pointers to decision tree subtrees. 
  shared_ptr<decision_tree_node> children[BTREE_ORDER + 1];
};

class rf{
public:
// No code in constructor or destructor
  rf();

  ~rf();

// Making rf_root a public variable
  shared_ptr<rf_btree> rf_root;
  shared_ptr<rf_btree> initiate_rf_root();
  
// Call "make_random_dt" as many times as you want to make random forest.  
// Here we have defined BTREE as 100, so there will be 100 trees in the random forest.
// We attach each random dt as a leaf.
  shared_ptr<rf_btree> buildrandomforest();


private:

  
};

#endif // RANDOMFOREST_H__