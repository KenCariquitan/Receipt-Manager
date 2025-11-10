import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'auth_gate.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Supabase.initialize(
    url: 'https://xadallnhdkafulcblcta.supabase.co',
    anonKey:
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhZGFsbG5oZGthZnVsY2JsY3RhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc5MzYxODAsImV4cCI6MjA3MzUxMjE4MH0.BTfMMZo5llL4X29apdrdvuEShdg5i_uo79xbWqw7E4Y',
  );
  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Receipt OCR',
      theme: ThemeData(useMaterial3: true),
      home: const AuthGate(), // decides login vs app
    );
  }
}

class AppConfig {
  static const apiBaseUrl = 'http://192.168.100.8:8000';
}
