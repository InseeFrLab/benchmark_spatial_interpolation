library(DBI)
library(duckdb)
library(dplyr)
library(glue)

conn_ddb <- DBI::dbConnect(duckdb::duckdb())

dbExecute(conn_ddb, "LOAD httpfs;")

dbExecute(
  conn_ddb,
  "
  CREATE OR REPLACE VIEW bdalti AS
  SELECT *
  FROM read_parquet('s3://oliviermeslin/BDALTI/BDALTI_parquet/**/*.parquet')
  "
)

bdalti <- tbl(conn_ddb, "bdalti") 


dbExecute(
  conn_ddb,
  "
  CREATE OR REPLACE VIEW rgealti AS
  SELECT *
  FROM read_parquet('s3://oliviermeslin/RGEALTI/RGEALTI_parquet/**/*.parquet')
  "
)

rgealti <- tbl(conn_ddb, "rgealti") 

bdalti  |> summarise(nb_bdalti = n() / 1e6, .by = departement) |>
  left_join(
rgealti |> summarise(nb_rgealti = n() / 1e6, .by = departement),
by = "departement"
) |> 
  mutate(ratio = nb_rgealti / nb_bdalti) |> 
  arrange(departement)
