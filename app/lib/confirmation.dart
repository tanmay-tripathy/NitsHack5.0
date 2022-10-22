import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

class ConfirmPage extends StatelessWidget {
  const ConfirmPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Lottie.asset('assets/tt.json', repeat: true),
          const Text(
            'Huray !!',
            style: TextStyle(fontSize: 22),
          ),
          SizedBox(
            height: 15,
          ),
          const Text(
            'Another step towards a green campus',
            style: TextStyle(fontSize: 18),
          )
        ],
      ),
    );
  }
}
