from fetch import *
from preprocess import *
from model import predict
import tensorflow as tf


def classify_object(oid, model, broker, max_len=50):
    df = find_file_local(oid, broker=broker)
    df = clean_and_rename(df)
    
    ra, dec = df["ra"].iloc[0], df["dec"].iloc[0]
    ebv = get_ebv(ra, dec)

    df, filter_cols = add_filters(df)
    df = combine_filters(df, filter_cols)
    
    df = ebv_correct(df, filter_cols, ebv)
    df = cap_long_seq(df, max_len=max_len)

    try:
        X = to_model_input(df, filter_cols, max_seq_len=max_len)
        return predict(model, X)
    except (ValueError, tf.errors.InvalidArgumentError) as e:
        if "shape" not in str(e).lower() and "expected" not in str(e).lower():
            raise  # not a shape issue — don't mask it
        X = to_model_input_with_time(df, filter_cols, max_seq_len=max_len)
        return predict(model, X)