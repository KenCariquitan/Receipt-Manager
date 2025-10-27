import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_image_compress/flutter_image_compress.dart';

import '../api_client.dart';
import '../models.dart';
import '../widgets/receipt_card.dart';

final apiProvider = Provider<ApiClient>((ref) => ApiClient());

class UploadPage extends ConsumerStatefulWidget {
  const UploadPage({super.key});
  @override
  ConsumerState<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends ConsumerState<UploadPage> {
  final _picker = ImagePicker();
  UploadResult? result;
  bool loading = false;
  String? error;
  String? selectedLabel;

  Future<void> _pick(ImageSource source) async {
    setState(() {
      loading = true;
      error = null;
      result = null;
    });

    try {
      // Compress at pick time (quality + resize)
      final picked = await _picker.pickImage(
        source: source,
        imageQuality: 82, // 0-100, lower = smaller size
        maxWidth: 1600,
        maxHeight: 1600,
      );
      if (picked == null) {
        setState(() => loading = false);
        return;
      }

      final preparedFile = await _ensureUnderLimit(picked);

      final sizeKb = (await preparedFile.length()) / 1024;
      // ignore: avoid_print
      print(
          "Prepared file: ${preparedFile.path}, size=${sizeKb.toStringAsFixed(1)} KB");

      // Wrap in multipart for upload
      final file = await http.MultipartFile.fromPath('file', preparedFile.path,
          filename: picked.name);

      final api = ref.read(apiProvider);
      final res = await api.uploadReceipt(file);
      setState(() {
        result = res;
        selectedLabel = res.category;
      });
    } catch (e) {
      setState(() {
        error = e.toString();
      });
    } finally {
      setState(() {
        loading = false;
      });
    }
  }

  Future<File> _ensureUnderLimit(XFile picked) async {
    const maxBytes = 1024 * 1024; // 1 MB hard cap
    final original = File(picked.path);
    if (!await original.exists()) {
      return original;
    }

    var current = original;
    var length = await current.length();
    if (length <= maxBytes) return current;

    final qualities = <int>[85, 75, 65, 55, 45, 35, 25, 20];
    File best = current;

    for (final quality in qualities) {
      final targetPath = _deriveCompressedPath(original.path, quality);
      final targetFile = File(targetPath);
      if (await targetFile.exists()) {
        await targetFile.delete();
      }
      final compressed = await FlutterImageCompress.compressAndGetFile(
        original.path,
        targetPath,
        quality: quality,
        minWidth: 1600,
        minHeight: 1600,
      );
      if (compressed == null) {
        continue;
      }
      final candidate = File(compressed.path);
      final candidateSize = await candidate.length();
      best = candidate;
      if (candidateSize <= maxBytes) {
        break;
      }
    }

    return best;
  }

  String _deriveCompressedPath(String originalPath, int quality) {
    final dot = originalPath.lastIndexOf('.');
    final suffix = '_cmp$quality';
    if (dot != -1) {
      return originalPath.substring(0, dot) +
          suffix +
          originalPath.substring(dot);
    }
    return '$originalPath$suffix.jpg';
  }

  Future<void> _saveCorrection() async {
    if (result == null || selectedLabel == null) return;
    try {
      setState(() => loading = true);
      final api = ref.read(apiProvider);

      // Update DB
      await api.updateReceipt(id: result!.id, category: selectedLabel);

      // Optional: also send feedback to improve ML
      await api.sendFeedback(text: result!.text, trueLabel: selectedLabel!);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Saved and sent feedback.')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    } finally {
      setState(() => loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: loading ? null : () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.photo_camera),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed:
                        loading ? null : () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (loading) const LinearProgressIndicator(),
            if (error != null)
              Text(error!, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 12),
            if (result != null)
              Expanded(
                child: ListView(
                  children: [
                    ReceiptCard(
                      res: result!,
                      selectedCategory: selectedLabel,
                      onCategoryChanged: (v) =>
                          setState(() => selectedLabel = v),
                    ),
                    const SizedBox(height: 12),
                    FilledButton(
                      onPressed: loading ? null : _saveCorrection,
                      child: const Text('Save Correction / Confirm'),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
