// Home page screen

import 'package:app/qrscanner.dart';
import 'package:app/signin_screen.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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
      // In body text containing 'Home page ' in center
      body: const Center(
        child: Text('Nothing to show here'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(context,
              MaterialPageRoute(builder: (context) => const QRScanner()));
        },
        child: const Icon(Icons.qr_code_scanner),
      ),
    );
  }
}
