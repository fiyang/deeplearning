// Streaming parser for original Ali-CCP sparse features. Never expands common rows.
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
const array<string,23> fields={"101","109_14","110_14","127_14","150_14","121","122","124","125","126","127","128","129","205","206","207","210","216","508","509","702","853","301"};
struct Feature{uint32_t token;float weight;};
using Bags=array<vector<Feature>,23>;
vector<string> prefix(const string&s,int n,size_t&pos){vector<string>v;pos=0;for(int i=0;i<n;i++){auto e=s.find(',',pos);if(e==string::npos)throw runtime_error("missing CSV columns");v.push_back(s.substr(pos,e-pos));pos=e+1;}return v;}
uint64_t number(const string&s){size_t used=0;auto n=stoull(s,&used);if(used!=s.size())throw runtime_error("invalid integer");return n;}
void parse_features(const string&s,size_t pos,uint64_t expected,Bags&bags){
 static unordered_map<string,int>fi=[](){unordered_map<string,int>m;for(int i=0;i<23;i++)m[fields[i]]=i;return m;}();
 for(auto&b:bags)b.clear();uint64_t count=0;
 while(pos<s.size()){
  auto end=s.find('\1',pos);if(end==string::npos)end=s.size();if(end==pos)throw runtime_error("empty feature");
  auto a=s.find('\2',pos),b=s.find('\3',pos);if(a>=end||b>=end||a>=b)throw runtime_error("invalid feature separators");
  auto f=fi.find(s.substr(pos,a-pos));if(f==fi.end())throw runtime_error("unknown field "+s.substr(pos,a-pos));
  uint64_t id=number(s.substr(a+1,b-a-1));if(id>(numeric_limits<uint32_t>::max()-22)/23)throw runtime_error("feature id overflow");
  size_t used=0;auto w=stof(s.substr(b+1,end-b-1),&used);if(used!=end-b-1||!isfinite(w))throw runtime_error("invalid feature weight");
  bags[f->second].push_back({uint32_t(id*23+f->second),w});count++;pos=end+1;
 }
 if(count!=expected)throw runtime_error("feature count differs from declared count");
}
struct Writer{
 ofstream ids,weights,offsets;uint64_t rows=0,tokens=0;array<uint64_t,23>totals{},maxlen{};uint32_t max_token=0;double minw=numeric_limits<double>::infinity(),maxw=-numeric_limits<double>::infinity();
 Writer(const string&p):ids(p+".tokens.bin",ios::binary),weights(p+".weights.bin",ios::binary),offsets(p+".offsets.bin",ios::binary){uint32_t z=0;offsets.write((char*)&z,4);if(!ids||!weights||!offsets)throw runtime_error("cannot create output");}
 void append(const Bags&bags){vector<uint32_t>ti;vector<float>tw;array<uint32_t,23>ends;for(int f=0;f<23;f++){totals[f]+=bags[f].size();maxlen[f]=max(maxlen[f],uint64_t(bags[f].size()));for(auto x:bags[f]){ti.push_back(x.token);tw.push_back(x.weight);max_token=max(max_token,x.token);minw=min(minw,double(x.weight));maxw=max(maxw,double(x.weight));}tokens+=bags[f].size();if(tokens>numeric_limits<uint32_t>::max())throw runtime_error("CSR offset exceeds uint32 capacity");ends[f]=tokens;}
 ids.write((char*)ti.data(),ti.size()*4);weights.write((char*)tw.data(),tw.size()*4);offsets.write((char*)ends.data(),23*4);rows++;if(!ids||!weights||!offsets)throw runtime_error("write failed");}
 void finish(){ids.close();weights.close();offsets.close();if(!ids||!weights||!offsets)throw runtime_error("flush failed");}
 void report(ostream&o){o<<"{\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"max_token\":"<<max_token<<",\"min_weight\":"<<(tokens?minw:0)<<",\"max_weight\":"<<(tokens?maxw:0)<<",\"field_tokens\":[";for(int i=0;i<23;i++)o<<(i?",":"")<<totals[i];o<<"],\"field_max_lengths\":[";for(int i=0;i<23;i++)o<<(i?",":"")<<maxlen[i];o<<"]}";}
};
int main(int argc,char**argv){try{
 if(argc!=4)throw runtime_error("usage: prepare common.csv skeleton.csv NEW_OUTPUT_DIR");string out=argv[3];if(filesystem::exists(out))throw runtime_error("output directory already exists");filesystem::create_directories(out);
 unordered_map<string,uint32_t>common;Bags bags;string line;size_t pos;ifstream cf(argv[1]),sf(argv[2]);if(!cf||!sf)throw runtime_error("cannot open source files");Writer cw(out+"/common"),sw(out+"/sample");
 while(getline(cf,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();auto v=prefix(line,2,pos);if(cw.rows>=numeric_limits<uint32_t>::max())throw runtime_error("too many common rows");if(!common.emplace(v[0],cw.rows).second)throw runtime_error("duplicate common id");parse_features(line,pos,number(v[1]),bags);cw.append(bags);if(cw.rows%100000==0)cout<<"COMMON rows="<<cw.rows<<" tokens="<<cw.tokens<<endl;}
 if(!cf.eof()||!cw.rows)throw runtime_error("common input read failed");cw.finish();ofstream refs(out+"/common_refs.bin",ios::binary),labels(out+"/labels.bin",ios::binary);array<uint64_t,4>pairs{};
 while(getline(sf,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();auto v=prefix(line,5,pos);auto c=number(v[1]),y=number(v[2]);if(c>1||y>1)throw runtime_error("nonbinary label");auto it=common.find(v[3]);if(it==common.end())throw runtime_error("missing referenced common id");parse_features(line,pos,number(v[4]),bags);sw.append(bags);auto idx=it->second;refs.write((char*)&idx,4);uint8_t pair[2]={uint8_t(c),uint8_t(y)};labels.write((char*)pair,2);pairs[c*2+y]++;if(sw.rows%1000000==0)cout<<"SAMPLE rows="<<sw.rows<<" tokens="<<sw.tokens<<endl;}
 if(!sf.eof()||!sw.rows)throw runtime_error("sample input read failed");sw.finish();refs.close();labels.close();if(!refs||!labels)throw runtime_error("label write failed");ofstream report(out+"/manifest.json");report<<"{\"format_version\":1,\"fields\":[";for(int i=0;i<23;i++)report<<(i?",":"")<<'"'<<fields[i]<<'"';report<<"],\"common\":";cw.report(report);report<<",\"sample\":";sw.report(report);report<<",\"label_pairs_00_01_10_11\":[";for(int i=0;i<4;i++)report<<(i?",":"")<<pairs[i];report<<"],\"weights_preserved\":true,\"features_truncated\":0,\"sample_order\":\"source line order\"}\n";report.close();if(!report)throw runtime_error("manifest write failed");cout<<"PARSE_COMPLETE "<<out<<endl;
 }catch(const exception&e){cerr<<"FAILED: "<<e.what()<<endl;return 1;}}
