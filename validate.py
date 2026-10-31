import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

# 1. Create a context and load the data
context = gx.get_context()
df = pd.read_csv("data/raw/students.csv")

# 2. Connect the dataset (data source -> asset -> batch definition)
data_source = context.data_sources.add_pandas("students_source")
asset = data_source.add_dataframe_asset("students_asset")
batch_def = asset.add_batch_definition_whole_dataframe("students_batch")
batch = batch_def.get_batch(batch_parameters={"dataframe": df})

# 3. Create the Expectation Suite
suite = context.suites.add(gx.ExpectationSuite(name="students_suite"))

# Expectation 1: student_id has no nulls
suite.add_expectation(
    gxe.ExpectColumnValuesToNotBeNull(column="student_id")
)

# Expectation 2: score is between 0 and 100
suite.add_expectation(
    gxe.ExpectColumnValuesToBeBetween(column="score", min_value=0, max_value=100)
)

# 4. Execute the validation
result = batch.validate(suite)
print("Overall success:", result.success)
for r in result.results:
    print(r.expectation_config.type, "->", r.success)