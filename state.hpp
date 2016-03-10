#ifndef STATE_H
#define STATE_H

//#include <set.h>

#include "utils.hpp"
#include "corpus.hpp"
#include "stirln.hpp"

using namespace std;

#define INIT_SIZE 100
#define LAZY_CHECK 5000
#define TOPIC_THRESHOLD 1.95

struct WordInfo {
public:
  int word_;
  int count_;
  int topic_assignment_;
};


class DocState {
public:
  int doc_id_;
  WordInfo* words_;
  int doc_length_;
public:
  DocState();
  ~DocState();
public:
  void setup_state_from_doc(const Document* doc);
};

//RhoMatrix is designed to be read-only and thus immutable.
//You can pass around pointers to it without fear!
class RhoMatrix {
private:
  size_t size;
  double** matrix;
  bool is_unit_matrix;
  double* cache_row;

public:
  RhoMatrix(size_t vocab_size);
  RhoMatrix(const char* filename);
  RhoMatrix(FILE* fileptr);
  ~RhoMatrix();
  bool check(size_t vocab_size); 
  bool isUnitMatrix();
  int get_vocab_size();
  int write_to_file(FILE* fileptr);
  void write_to_file(const char* filename);
  inline double  get_element(size_t row_idx, size_t col_idx);
  inline double* get_row(size_t word_idx);
private:
  void make_unit_matrix(size_t vocab_size);
  void read_from_file(const char* filename);
  int read_from_file(FILE* fileptr);
  void free_and_reset();
  void init();
};

class HDPState {
public:
  HDPState();
  ~HDPState();
  void init_hdp_state(double eta, double gamma, double alpha, int size_vocab, const char* rho_matrix_fn);
  void copy_hdp_state(const HDPState& src_state);
  void compact_hdp_state(vct_int* k_to_new_k);
  void save_hdp_state(const char* name);
  void load_hdp_state(const char* name);
public:
  vector<int* > topic_lambda_; // Not counting the prior, eta
  vct_int word_counts_by_topic_;
  vct_int beta_u_; // Not counting the prior 1.0
  //vct beta_v_; // Not counting the piror gamma
  vct pi_;     // A sample of pi.
  double pi_left_;
  RhoMatrix* rho_matrix_;
  char* rho_matrix_filename;
  vct_int topic_rho_assignments_;

  // Hyper parameters
  double eta_;
  double gamma_;
  double alpha_;

  int num_topics_;
  int size_vocab_;
};

class HDP {
public: // Not fixed.
  int num_docs_;
  DocState** doc_states_;

  vector<int*> word_counts_by_topic_doc_;  // Number of words by topic, doc
  vector<int*> table_counts_by_topic_doc_; // Number of tables by topic, doc.

  HDPState* hdp_state_;

  // For fast Gibbs sampling using Mimno's trick.
  vector<set<int> > unique_topic_by_word_;
  vector<set<int> > unique_topic_by_doc_;
  vct smoothing_prob_;
  double smoothing_prob_sum_;
  vector<double* > doc_prob_;
  vct doc_prob_sum_;

public:
  Stirling stirling_;

public:
  HDP();
  ~HDP();

  void init_hdp(double eta, double gamma, double alpha, int size_vocab, const char* rho_matrix_fn);
  void setup_doc_states(const vector<Document* >& docs);
  void remove_doc_states();

  void compact_hdp_state();
  void updateTopicRhoAssignments();
  int iterate_gibbs_state(bool remove, bool permute);
  int  sample_word_assignment(DocState* doc_state, int i, bool remove, vct* p);
  void doc_state_update(DocState* doc_state, int i, int update);
  void sample_table_counts(DocState* doc_state, vct* p);
  void sample_top_level_proportions();

  double log_likelihood(const HDPState* old_hdp_state = NULL);
  void save_state(const char* name);
  void load_state(const char* name);
  void hyper_inference(double gamma_a, double gamma_b, double alpha_a, double alpha_b);

  void init_fast_gibbs_sampling_variables();
  void save_doc_states(const char* name);
  void sample_first_level_concentration(double gamma_a, double gamma_b);
  void sample_second_level_concentration(double alpha_a, double alpha_b);
};




#endif // STATE_H
