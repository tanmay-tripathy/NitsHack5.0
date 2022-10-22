import 'dart:convert';
import 'dart:io';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:qr_code_scanner/qr_code_scanner.dart';
import 'package:http/http.dart' as http;

class QRScanner extends StatefulWidget {
  const QRScanner({Key? key}) : super(key: key);

  @override
  State<StatefulWidget> createState() => _QRScannerState();
}

class _QRScannerState extends State<QRScanner> {
  final GlobalKey qrKey = GlobalKey(debugLabel: 'QR');
  Barcode? result;
  QRViewController? controller;

  // In order to get hot reload to work we need to pause the camera if the platform
  // is android, or resume the camera if the platform is iOS.
  @override
  void reassemble() {
    super.reassemble();
    if (Platform.isAndroid) {
      controller!.pauseCamera();
    } else if (Platform.isIOS) {
      controller!.resumeCamera();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: <Widget>[
          Expanded(
            flex: 5,
            child: QRView(
              key: qrKey,
              onQRViewCreated: _onQRViewCreated,
              overlay: QrScannerOverlayShape(
                borderColor: Colors.red,
                borderRadius: 10,
                borderLength: 30,
                borderWidth: 10,
                cutOutSize: 300,
              ),
            ),
          ),
          Expanded(
            flex: 1,
            child: Center(
              child: (result != null)
                  ? Text('Dusbin ID: ${result!.code}')
                  : Text('Scan a code'),
            ),
          )
        ],
      ),
    );
  }

  void _onQRViewCreated(QRViewController controller) {
    this.controller = controller;
    controller.scannedDataStream.listen((scanData) {
      setState(() {
        result = scanData;
      });
      postRequest(result?.code);
    });
  }

  void postRequest(String? id) async {
    try {
      final Uri getUri = Uri.parse('https://nitshack.herokuapp.com/user/');
      final http.Request getRequest = http.Request('POST', getUri);
      getRequest.headers
          .addAll({'Origin': '*', 'Access-Control-Allow-Origin': '*'});

      String auth = FirebaseAuth.instance.currentUser!.uid;
      Map<String, String> body = {'userID': auth.toString(), 'dustBin': id!};
      getRequest.body = json.encode(body.toString());

      final http.StreamedResponse response = await getRequest.send();
      final String responseString = await response.stream.bytesToString();

      final Map<String, dynamic> responseBody =
          json.decode(responseString) as Map<String, dynamic>;
      print(responseBody.toString());
    } catch (e) {
      print(e.toString());
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }
}
