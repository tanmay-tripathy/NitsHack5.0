// Home page screen

import 'dart:convert';

import 'package:app/model/historyModel.dart';
import 'package:app/qrscanner.dart';
import 'package:app/signin_screen.dart';
import 'package:app/utils/loader.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class HomePage extends StatefulWidget {
  HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool loading = true;
  Future<List<HistoryModel>> fetchdata() async {
    String auth = FirebaseAuth.instance.currentUser!.uid;
    //dummy url for testing
    final response = await http.get(
      Uri.parse('https://nitshack.herokuapp.com/user/dsa321dsa'),
      headers: {'Origin': '*', 'Access-Control-Allow-Origin': '*'},
    );
    print(response.body);
    print((response.body[0].length));
    List<HistoryModel> history = [];

    for (var i = 0; i < response.body[0].length; i++) {
      print(i);
      history.add(HistoryModel.fromJson(jsonDecode(response.body)[i]));
    }
    print(history);
    setState(() {
      loading = false;
    });
    return history;
  }

  @override
  initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.green,
        centerTitle: true,
        actions: <Widget>[
          Padding(
              padding: EdgeInsets.only(right: 20.0),
              child: GestureDetector(
                onTap: () {
                  FirebaseAuth auth = FirebaseAuth.instance;
                  auth.signOut();
                  Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(builder: (context) => SignInScreen()),
                      (route) => false);
                },
                child: const Icon(
                  Icons.logout,
                  size: 26.0,
                ),
              )),
        ],

        // on appbar text containing 'GEEKS FOR GEEKS'
        title: const Text("History"),
      ),
      body: Container(
        padding: EdgeInsets.all(16.0),
        child: FutureBuilder(
          future: fetchdata(),
          builder: (BuildContext ctx, AsyncSnapshot snapshot) {
            if (snapshot.data == null) {
              return Container(
                child: Center(
                  child: CircularProgressIndicator(),
                ),
              );
            } else {
              return ListView.builder(
                  itemCount: snapshot.data.length,
                  itemBuilder: (ctx, index) => Container(
                        margin: const EdgeInsets.symmetric(
                            vertical: 10, horizontal: 5),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: <Color>[
                              Color.fromARGB(255, 143, 210, 255)
                                  .withOpacity(0.15),
                              Color.fromARGB(255, 132, 245, 255)
                                  .withOpacity(0.15),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                              color: const Color.fromARGB(255, 105, 218, 249)
                                  .withOpacity(0.48),
                              width: 2),
                        ),
                        child: ListTile(
                            minVerticalPadding: 10,
                            title: Padding(
                              padding: const EdgeInsets.only(top: 10),
                              child: Text(
                                'Weight dumped: ${snapshot.data[index].weight} kg',
                                style: const TextStyle(
                                    fontWeight: FontWeight.bold),
                              ),
                            ),
                            tileColor: Colors.transparent,
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: <Widget>[
                                Text(
                                  'Biodegradable composition: ${snapshot.data[index].biodegradable} %',
                                  style: const TextStyle(fontSize: 12),
                                ),
                                const SizedBox(
                                  height: 3,
                                ),
                                Text(
                                    'Non-biodegradable composition: ${snapshot.data[index].nonbiodegradable} %',
                                    style: const TextStyle(fontSize: 12)),
                              ],
                            ),
                            trailing: const Chip(
                              backgroundColor:
                                  Color.fromARGB(255, 149, 245, 152),
                              label: Text(
                                '22th Oct',
                                style: TextStyle(
                                    fontSize: 12, color: Colors.black),
                              ),
                            )),
                      ));
            }
          },
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => const QRScanner(),
            ),
          );
        },
        child: const Icon(Icons.qr_code_scanner),
      ),
    );
  }
}
