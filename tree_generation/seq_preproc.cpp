#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdint>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

//#include "seq_preproc.hpp"
#include "tree.hpp"

struct ColumnInfo {
    size_t len;
    size_t id;
    std::string pattern;

    bool operator<(const ColumnInfo& other) const { return pattern < other.pattern; }
    bool operator==(const ColumnInfo& other) const { return pattern == other.pattern; }

};

void remove_sparse_columns(
    std::vector<std::string>& rows,
    std::vector<NewPlacementQuery>& queries,
    size_t& sites,
    double gap_threshold = 0.8
) {
    size_t total_sequences = rows.size() + queries.size();
    std::vector<int> filtered_mask(sites, 0);
    size_t filtered_sites = 0;

    tbb::parallel_for(tbb::blocked_range<size_t> (0, sites), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t col = r.begin(); col < r.end(); col++) {
            size_t gap_count = 0;
            for(const auto& row : rows) {
                if(row[col] == '-') {
                    gap_count++;
                }
            }
            for(const auto& q : queries) {
                if(q.msa[col] == '-') {
                    gap_count++;
                }
            }
            double gap_ratio = static_cast<double> (gap_count) / total_sequences;

            if(gap_ratio < 0.7) {
                filtered_mask[col] = 1;
            }
        }
    });

    for(int m : filtered_mask) {
        if(m) {
            filtered_sites++;
        }
    }
    std::vector<std::string> new_rows(rows.size());
    printf("filtered_sites = %d\n", filtered_sites);

    tbb::parallel_for(tbb::blocked_range<size_t> (0, rows.size()), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i = r.begin(); i < r.end(); i++) {
            std::string filtered;
            filtered.reserve(filtered_sites);
            for(size_t col = 0; col < sites; col++) {
                if(filtered_mask[col]) {
                    filtered += rows[i][col];
                 }
             }
             new_rows[i] = std::move(filtered);
        }
    });
    printf("%s\n\n", rows[0].c_str());
    printf("%s\n\n", new_rows[0].c_str());


    tbb::parallel_for(tbb::blocked_range<size_t> (0, queries.size()), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i = r.begin(); i < r.end(); i++) {
            std::string filtered;
            filtered.reserve(filtered_sites);
            for(size_t col = 0; col < sites; col++) {
                if(filtered_mask[col]) {
                    filtered += queries[i].msa[col];
                 }
             }
             queries[i].msa = std::move(filtered);
        }
    });

    rows = std::move(new_rows);
    sites = filtered_sites;
}

void remove_repetitive_columns(
    std::vector<std::string>& rows,
    std::vector<NewPlacementQuery>& queries,
    size_t& sites
) {
    size_t total_sequences = rows.size() + queries.size();
    std::vector<ColumnInfo> columns(sites);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sites), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t col = r.begin(); col != r.end(); col++) {
            std::string pattern;
            pattern.reserve(total_sequences);
            for (const auto& row : rows) pattern += row[col];
            for (const auto& q : queries) pattern += q.msa[col];

            columns[col].pattern = std::move(pattern);
            columns[col].len     = total_sequences;
            columns[col].id      = col;
            //printf("%s\n\n", columns[col].pattern.c_str());
        }
    });

    //printf("-----------------------\n\n");
    //printf("before\n");
    //for(auto & r : columns) {
    //    printf("%d\t%s\n", r.id, r.pattern.c_str());
    //}
    tbb::parallel_sort(columns.begin(), columns.end());
    //printf("-----------------------\n\n");
    //printf("sorted\n");
    //int i = 0;
    //for(auto & r : columns) {
    //    printf("%d\t%d\t%s\n", i, r.id, r.pattern.c_str());
    //    i++;
    //}

    std::vector<size_t> unique_id(sites);
    unique_id[0] = columns[0].id;
    size_t filtered_sites = 1;
    
    for(size_t i = 1; i < sites; i++) {
        if(!(columns[i] == columns[i-1])) {
            //printf("push back: %d, id=%d, sites2=%d\n", i, columns[i].id, filtered_sites2);
            unique_id[filtered_sites] = columns[i].id;
            filtered_sites++;
            //printf("filtered_sites2: %d\n", filtered_sites2);
        }
    }
    printf("filtered_sites = %d\n", filtered_sites);



    std::sort(unique_id.begin(), unique_id.begin() + filtered_sites);

    //printf("unique_id: \n");
    //for(size_t i = 0; i < filtered_sites; i++)
    //    printf("i=%d, %d\n", i, unique_id[i]);

    std::vector<std::string> new_rows(rows.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, rows.size()), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i = r.begin(); i < r.end(); i++) {
            std::string pattern;
            pattern.reserve(filtered_sites);
            //for(size_t col: unique_id) {
            for(size_t j = 0; j < filtered_sites; j++) {
                size_t col = unique_id[j];
                pattern += rows[i][col];
                //if(i == 0) printf("col=%d, pattern=%c\n\n", col, rows[i][col]);
            }
            new_rows[i] = std::move(pattern);
        }
    });
    printf("%s\n\n", rows[0].c_str());
    printf("%s\n\n", new_rows[0].c_str());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, queries.size()), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i = r.begin(); i < r.end(); i++) {
            std::string pattern;
            pattern.reserve(filtered_sites);
            //for(size_t col: unique_id) {
            for(size_t j = 0; j < filtered_sites; j++) {
                size_t col = unique_id[j];
                pattern += queries[i].msa[col];
            }
            queries[i].msa = std::move(pattern);
        }
    });

    rows = std::move(new_rows);
    sites = filtered_sites;

}
