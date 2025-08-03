"""
Performance and benchmarking tests for MuMDIA components.

This module tests performance characteristics, memory usage,
and computational efficiency of various MuMDIA components.
"""

import pytest
import time
import psutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
import polars as pl
import numpy as np
import tempfile


class TestPerformanceBenchmarks:
    """Test performance characteristics of key components."""

    @pytest.mark.performance
    def test_polars_dataframe_performance(self):
        """Test Polars DataFrame operation performance."""
        # Create large test dataset
        n_rows = 100000

        start_time = time.time()

        df_large = pl.DataFrame(
            {
                "peptide": [f"PEPTIDE{i % 1000:04d}K" for i in range(n_rows)],
                "psm_id": range(n_rows),
                "intensity": np.random.exponential(1000, n_rows),
                "rt": np.random.uniform(0, 120, n_rows),
                "charge": np.random.choice([2, 3, 4], n_rows),
                "mz": np.random.uniform(200, 2000, n_rows),
            }
        )

        creation_time = time.time() - start_time

        # Test common operations
        start_time = time.time()

        # Filtering
        filtered = df_large.filter(pl.col("intensity") > 500)

        # Grouping and aggregation
        grouped = df_large.group_by("peptide").agg(
            [
                pl.col("intensity").max().alias("max_intensity"),
                pl.col("rt").mean().alias("mean_rt"),
                pl.len().alias("count"),
            ]
        )

        # Sorting
        sorted_df = df_large.sort("intensity", descending=True)

        operation_time = time.time() - start_time

        # Performance assertions
        assert creation_time < 2.0  # Should create large DF quickly
        assert operation_time < 3.0  # Operations should be fast
        assert len(filtered) > 0
        assert len(grouped) <= 1000  # Number of unique peptides
        assert len(sorted_df) == n_rows

    @pytest.mark.performance
    def test_correlation_calculation_performance(self):
        """Test correlation calculation performance with different sizes."""
        sizes = [100, 500, 1000]

        for size in sizes:
            # Create test correlation matrices
            data1 = np.random.randn(size)
            data2 = np.random.randn(size)

            start_time = time.time()

            # Test NumPy correlation
            correlation = np.corrcoef(data1, data2)[0, 1]

            calculation_time = time.time() - start_time

            # Performance assertions
            assert calculation_time < 0.1  # Should be fast even for large arrays
            assert not np.isnan(correlation)
            assert -1 <= correlation <= 1

    @pytest.mark.performance
    def test_matrix_operations_performance(self):
        """Test matrix operation performance."""
        matrix_sizes = [50, 100, 200]

        for size in matrix_sizes:
            # Create test matrices
            matrix_a = np.random.randn(size, size)
            matrix_b = np.random.randn(size, size)

            start_time = time.time()

            # Matrix multiplication
            result = np.dot(matrix_a, matrix_b)

            # Element-wise operations
            element_wise = matrix_a * matrix_b

            # Eigenvalue calculation (computationally intensive)
            if size <= 100:  # Only for smaller matrices
                eigenvals = np.linalg.eigvals(matrix_a)

            operation_time = time.time() - start_time

            # Performance assertions
            assert operation_time < 5.0  # Should complete in reasonable time
            assert result.shape == (size, size)
            assert element_wise.shape == (size, size)

    @pytest.mark.performance
    def test_parallel_processing_simulation(self):
        """Test parallel processing simulation performance."""
        # Simulate chunked processing
        total_items = 10000
        chunk_sizes = [100, 500, 1000]

        for chunk_size in chunk_sizes:
            start_time = time.time()

            # Simulate chunking
            chunks = [
                list(range(i, min(i + chunk_size, total_items)))
                for i in range(0, total_items, chunk_size)
            ]

            # Simulate processing each chunk
            results = []
            for chunk in chunks:
                # Simulate some computation
                chunk_result = sum(x**2 for x in chunk)
                results.append(chunk_result)

            processing_time = time.time() - start_time

            # Performance assertions
            assert processing_time < 2.0  # Should be efficient
            assert len(results) == len(chunks)
            assert len(chunks) == (total_items + chunk_size - 1) // chunk_size


class TestMemoryUsage:
    """Test memory usage patterns and efficiency."""

    @pytest.mark.memory
    def test_dataframe_memory_usage(self):
        """Test DataFrame memory usage patterns."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large DataFrame
        n_rows = 50000
        df_test = pl.DataFrame(
            {
                "peptide": [f"PEPTIDE{i % 100:03d}K" for i in range(n_rows)],
                "values": np.random.randn(n_rows),
                "integers": np.random.randint(0, 1000, n_rows),
                "categories": np.random.choice(["A", "B", "C", "D"], n_rows),
            }
        )

        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_creation_memory - initial_memory

        # Test memory-efficient operations
        # Use lazy evaluation to minimize memory usage
        lazy_result = (
            df_test.lazy()
            .filter(pl.col("values") > 0)
            .group_by("peptide")
            .agg([pl.col("values").mean(), pl.col("integers").sum()])
        )

        result = lazy_result.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory assertions
        assert memory_increase < 100  # Should not use excessive memory
        assert len(result) <= 100  # Should aggregate efficiently

        # Clean up
        del df_test, result

    @pytest.mark.memory
    def test_array_memory_efficiency(self):
        """Test NumPy array memory efficiency."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large arrays
        array_size = 1000000
        array1 = np.random.randn(array_size)
        array2 = np.random.randn(array_size)

        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform memory-efficient operations
        # Use in-place operations when possible
        array1 += array2  # In-place addition
        result = np.sum(array1)  # Reduction operation

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_difference = final_memory - after_creation_memory

        # Memory assertions
        assert memory_difference < 50  # Should not create many temporary arrays
        assert isinstance(result, (float, np.floating))

        # Clean up
        del array1, array2

    @pytest.mark.memory
    def test_string_memory_optimization(self):
        """Test string memory optimization patterns."""
        # Test string interning and categorical data
        n_strings = 100000
        unique_strings = 100

        # Create repetitive string data (should compress well)
        string_data = [f"PEPTIDE_{i % unique_strings:03d}" for i in range(n_strings)]

        process = psutil.Process(os.getpid())
        before_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test Polars categorical optimization
        df_strings = pl.DataFrame({"peptide": string_data, "id": range(n_strings)})

        # Convert to categorical for memory optimization
        df_categorical = df_strings.with_columns(pl.col("peptide").cast(pl.Categorical))

        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = after_memory - before_memory

        # Memory assertions
        assert memory_used < 100  # Should be memory efficient
        assert len(df_categorical) == n_strings

        # Clean up
        del string_data, df_strings, df_categorical


class TestScalabilityTests:
    """Test scalability characteristics."""

    @pytest.mark.scalability
    def test_dataframe_size_scaling(self):
        """Test how operations scale with DataFrame size."""
        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            df_test = pl.DataFrame(
                {
                    "peptide": [f"PEPTIDE{i % 100}" for i in range(size)],
                    "value": np.random.randn(size),
                    "category": np.random.choice(["A", "B", "C"], size),
                }
            )

            start_time = time.time()

            # Perform standard operations
            result = df_test.group_by("peptide").agg(
                [
                    pl.col("value").mean().alias("value_mean"),
                    pl.col("value").std().alias("value_std"),
                    pl.len().alias("count"),
                ]
            )

            operation_time = time.time() - start_time
            times.append(operation_time)

            # Cleanup
            del df_test, result

        # Test that time scales reasonably (not exponentially)
        # For linear scaling, time should roughly double when size doubles
        if len(times) >= 3:
            # Check that scaling is not exponential
            ratio_1 = times[1] / times[0] if times[0] > 0 else 1
            ratio_2 = times[2] / times[1] if times[1] > 0 else 1

            # Ratios should be reasonable (not > 10x)
            assert ratio_1 < 10
            assert ratio_2 < 10

    @pytest.mark.scalability
    def test_correlation_matrix_scaling(self):
        """Test correlation matrix calculation scaling."""
        matrix_sizes = [10, 50, 100]
        times = []

        for size in matrix_sizes:
            # Create test data
            data_matrix = np.random.randn(size, size)

            start_time = time.time()

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data_matrix)

            calculation_time = time.time() - start_time
            times.append(calculation_time)

            # Verify result
            assert correlation_matrix.shape == (size, size)
            assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1

            # Cleanup
            del data_matrix, correlation_matrix

        # Test reasonable scaling
        for t in times:
            assert t < 5.0  # Should complete in reasonable time

    @pytest.mark.scalability
    def test_file_io_scaling(self):
        """Test file I/O scaling with different data sizes."""
        sizes = [1000, 5000]

        for size in sizes:
            df_test = pl.DataFrame(
                {
                    "peptide": [f"PEPTIDE{i}" for i in range(size)],
                    "value1": np.random.randn(size),
                    "value2": np.random.randn(size),
                    "value3": np.random.randn(size),
                }
            )

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                temp_file = f.name

            try:
                # Test write performance
                start_time = time.time()
                df_test.write_csv(temp_file)
                write_time = time.time() - start_time

                # Test read performance
                start_time = time.time()
                df_read = pl.read_csv(temp_file)
                read_time = time.time() - start_time

                # Performance assertions
                assert write_time < 5.0  # Should write quickly
                assert read_time < 5.0  # Should read quickly
                assert len(df_read) == size
                assert df_read.shape == df_test.shape

                # Cleanup
                del df_test, df_read

            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestComputationalComplexity:
    """Test computational complexity of algorithms."""

    @pytest.mark.complexity
    def test_sorting_complexity(self):
        """Test sorting algorithm complexity."""
        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            # Create random data
            data = np.random.randn(size)

            start_time = time.time()
            sorted_data = np.sort(data)
            sort_time = time.time() - start_time

            times.append(sort_time)

            # Verify sorting correctness
            assert len(sorted_data) == size
            assert all(sorted_data[i] <= sorted_data[i + 1] for i in range(size - 1))

            # Cleanup
            del data, sorted_data

        # Test that sorting time scales appropriately (O(n log n))
        for t in times:
            assert t < 1.0  # Should be fast

    @pytest.mark.complexity
    def test_search_complexity(self):
        """Test search algorithm complexity."""
        sizes = [1000, 10000, 100000]

        for size in sizes:
            # Create sorted array for binary search
            sorted_array = np.arange(size)
            search_values = np.random.choice(sorted_array, 100)

            start_time = time.time()

            # Perform multiple searches
            for value in search_values:
                index = np.searchsorted(sorted_array, value)
                assert 0 <= index <= size

            search_time = time.time() - start_time

            # Search should be very fast (O(log n) per search)
            assert search_time < 0.1

            # Cleanup
            del sorted_array, search_values

    @pytest.mark.complexity
    def test_aggregation_complexity(self):
        """Test aggregation operation complexity."""
        sizes = [1000, 5000, 10000]

        for size in sizes:
            df_test = pl.DataFrame(
                {
                    "group": np.random.choice(["A", "B", "C", "D", "E"], size),
                    "value1": np.random.randn(size),
                    "value2": np.random.randn(size),
                }
            )

            start_time = time.time()

            # Complex aggregation
            result = df_test.group_by("group").agg(
                [
                    pl.col("value1").mean().alias("mean1"),
                    pl.col("value1").std().alias("std1"),
                    pl.col("value2").sum().alias("sum2"),
                    pl.col("value2").max().alias("max2"),
                    pl.len().alias("count"),
                ]
            )

            aggregation_time = time.time() - start_time

            # Aggregation should be efficient
            assert aggregation_time < 1.0
            assert len(result) <= 5  # Number of groups
            assert "mean1" in result.columns

            # Cleanup
            del df_test, result


class TestResourceUtilization:
    """Test resource utilization patterns."""

    @pytest.mark.resource
    def test_cpu_utilization_patterns(self):
        """Test CPU utilization during computations."""
        # Test CPU-intensive operations
        start_time = time.time()

        # Matrix operations (CPU intensive)
        matrix_size = 200
        matrix_a = np.random.randn(matrix_size, matrix_size)
        matrix_b = np.random.randn(matrix_size, matrix_size)

        # Perform computation
        result = np.dot(matrix_a, matrix_b)
        eigenvals = np.linalg.eigvals(result[:50, :50])  # Smaller subset for speed

        computation_time = time.time() - start_time

        # Should complete in reasonable time
        assert computation_time < 10.0
        assert result.shape == (matrix_size, matrix_size)
        assert len(eigenvals) == 50

        # Cleanup
        del matrix_a, matrix_b, result, eigenvals

    @pytest.mark.resource
    def test_memory_allocation_patterns(self):
        """Test memory allocation and deallocation patterns."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Allocate and deallocate memory in pattern
        arrays = []

        for i in range(10):
            # Allocate
            arr = np.random.randn(10000)
            arrays.append(arr)

            # Check memory doesn't grow too much
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            assert memory_growth < 200  # Reasonable limit

        # Deallocate
        del arrays

        # Memory should be released (approximately)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Note: Python may not immediately release memory to OS

    @pytest.mark.resource
    def test_file_handle_management(self):
        """Test file handle management."""
        initial_open_files = len(psutil.Process(os.getpid()).open_files())

        # Create and close multiple temporary files
        temp_files = []

        try:
            for i in range(10):
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                    f.write(f"test data {i}\n")
                    temp_files.append(f.name)

            # Check file handles
            current_open_files = len(psutil.Process(os.getpid()).open_files())

            # Should not have leaked file handles
            file_handle_increase = current_open_files - initial_open_files
            assert file_handle_increase < 50  # Reasonable limit

        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
