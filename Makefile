CC = g++
CFLAGS = -g -Wall -O3 -std=c++14
LIBS = -larmadillo
MODELS = time_svd baseline_predictor matrix_factor_bias
COMBINATION = grid_search

MODEL_OBJS = $(addsuffix .o,$(addprefix out/,$(MODELS)))

all: $(addprefix bin/,$(MODELS) $(COMBINATION))

# Build data processing object files in commmon with all models.
out/data.o: data_processing/data.cpp
	$(CC) $(CFLAGS) -c data_processing/data.cpp -o out/data.o

$(MODEL_OBJS): $(addsuffix .cpp,$(addprefix $(basename $(notdir $@))/,$(basename $(notdir $@)))) 
	$(CC) $(CFLAGS) -c $(addsuffix .cpp,$(addprefix $(basename $(notdir $@))/,$(basename $(notdir $@)))) -o $@

# Build models.
$(addprefix bin/,$(MODELS)): out/data.o $(MODEL_OBJS)
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^ $(addsuffix /train.cpp,$(notdir $@))

$(addprefix bin/,$(COMBINATION)): out/data.o $(MODEL_OBJS) $(addsuffix /*.cpp, $(notdir $(COMBINATION)))
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^ 

clean:
	rm -rf out/* bin/*


