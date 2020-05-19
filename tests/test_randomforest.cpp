// If you change anything in this file, your changes will be ignored 
// in your homework submission.
// Chekout TEST_F functions bellow to learn what is being tested.
#include <gtest/gtest.h>
#include "../code/randomforest.h"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

class test_randomforest: public ::testing::Test {
protected:
	// This function runs only once before any TEST_F function
	static void SetUpTestCase(){
	}

	// This function runs after all TEST_F functions have been executed
	static void TearDownTestCase(){
	}
    
	// this function runs before every TEST_F function
	void SetUp() override {
    }

	// this function runs after every TEST_F function
	void TearDown() override {
	}
};

TEST_F(test_randomforest, SanityCheck){
	data_class test_obj;
//test to see if constructor works
	ASSERT_EQ(test_obj.training_data.at(0).color, GREEN);
	ASSERT_EQ(test_obj.training_data.at(1).diameter, 3);
	ASSERT_EQ(test_obj.training_data.at(2).color, RED);
	ASSERT_EQ(test_obj.training_data.at(3).label, GRAPE);
	ASSERT_EQ(test_obj.training_data.at(4).label, LEMON);
	ASSERT_TRUE(true);
//   ASSERT_TRUE(private_contains(small, 13));
//   ASSERT_FALSE(private_contains(small, 14));

//   shared_ptr<btree> broken = build_broken(); // invariant should fail
//   ASSERT_FALSE(check_tree(broken)); // be sure we catch that
}

TEST_F(test_randomforest, CalculateMatches){
	data_class test_obj;
	vector<data_node>training_data = test_obj.training_data;

//test to see if function works
	ASSERT_EQ(test_obj.calculate_matches(test_obj.training_data, "color", YELLOW), 2);
	ASSERT_EQ(test_obj.calculate_matches(test_obj.training_data, "label", LEMON), 1);
//   shared_ptr<btree> broken = build_broken(); // invariant should fail
//   ASSERT_FALSE(check_tree(broken)); // be sure we catch that
}

TEST_F(test_randomforest, giniImpurity){
	data_class test_obj;
//test to see if function works
//ASSERT_NEAR (expected, actual, absolute_range)
	ASSERT_NEAR(test_obj.gini_Impurity(test_obj.training_data, "color", YELLOW), 0.48, .1);
	ASSERT_NEAR(test_obj.gini_Impurity(test_obj.training_data, "label", LEMON), 0.32, .1);
}

TEST_F(test_randomforest, testBootstrap){
	data_class test_obj;
//test size of function is 5
	ASSERT_EQ(test_obj.bootstrap(test_obj.training_data).size(), 5);
//test that the random vectors are not equal
	//ASSERT_NE(test_obj.Questions(test_obj.training_data).at(0), test_obj.Questions(test_obj.training_data).at(0));
}
// shared_ptr<decision_tree_node> root_node (new decision_tree_node);
//     // ii. create bootstrapped dataset
//     vector<data_node> bootstrapped_data = bootstrap(training_data);
//     root_node->_data_nodes = training_data;
//     // iii. calculate information gain for category 
//     int best_split = find_best_split(bootstrapped_data, category);
//     // iv. store gini impurity of information gain/ best split 
//     float gini_impurity = gini_Impurity(bootstrapped_data, category, best_split);
//     root_node->gini_impurity = gini_impurity;
//     // v. is_leaf is true
//     root_node->isLeaf = true;
//     // vi. create decision tree  shared pointer and load with training data
//     return root_node;
shared_ptr<decision_tree_node> Load_DT(vector<data_node>training_data, string category);
TEST_F(test_randomforest, testLoadDT){
	data_class test_obj;
//test size of function is 5
	vector<data_node> test_data;
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));
    
	shared_ptr<decision_tree_node> test_dt_node = test_obj.Load_DT(test_data, "color");
	ASSERT_EQ(test_dt_node->best_split, 10);
	ASSERT_EQ(test_dt_node->gini_impurity, 0);
	ASSERT_TRUE(test_dt_node->isLeaf);
	ASSERT_EQ(test_dt_node->_data_nodes.at(0).color, 10);
	ASSERT_EQ(test_dt_node->_data_nodes.at(0).diameter, 3);
	ASSERT_EQ(test_dt_node->_data_nodes.at(0).label, 40);

}

TEST_F(test_randomforest, testLoadSplit){
	data_class test_obj;
//test size of function is 5
	vector<data_node> test_data;
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,2, APPLE));
	test_data.push_back(data_node(GREEN,1, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));
	test_data.push_back(data_node(GREEN,3, APPLE));

    shared_ptr<decision_tree_node> test_dt_node = test_obj.Load_DT(test_data, "color");
	//what should happen is all the nodes should go to the right node
	shared_ptr<decision_tree_node> test_split_node = test_obj.load_Split(test_dt_node, test_data, test_dt_node->best_split, "color");
	ASSERT_EQ(test_split_node->right->_data_nodes.at(0).color, GREEN);
	ASSERT_EQ(test_split_node->right->_data_nodes.at(4).diameter, 3);
}

TEST_F(test_randomforest, testMakeRandomDT){
	data_class test_obj;
//test size of function is 5
	vector<data_node> test_data;
	test_data.push_back(data_node(GREEN,1, APPLE));
	test_data.push_back(data_node(GREEN,1, APPLE));
	test_data.push_back(data_node(GREEN,1, APPLE));
	test_data.push_back(data_node(GREEN,1, APPLE));
	test_data.push_back(data_node(GREEN,1, APPLE));

	shared_ptr<decision_tree_node> random_dt = test_obj.Make_Random_DT(test_data);
	//all the nodes should go to the right node
	
	ASSERT_EQ(random_dt->right->_data_nodes.at(0).color, GREEN);
	ASSERT_EQ(random_dt->right->_data_nodes.at(3).color, GREEN);
	

}

TEST_F(test_randomforest, testbuildrf){
	rf rf_obj;
	//shared_ptr<rf_btree> test_rf_root = rf_obj.initiate_rf_root();
	shared_ptr<rf_btree> test_rf_root = rf_obj.buildrandomforest();
	// cout << test_rf_root->children[0]->_data_nodes.at(0).color<< endl;
	// cout << test_rf_root->children[50]->_data_nodes.at(0).color<< endl;
	// cout << test_rf_root->children[99]->_data_nodes.at(0).color<< endl;

}
