
service Partition {
    string partition(1:map<string,string> file, 2:i16 ep, 3:i16 pp),
    map<string,map<string,list<double>>> load_server_regression_result(),
    list<i32> optimize_pp_ep(),
}