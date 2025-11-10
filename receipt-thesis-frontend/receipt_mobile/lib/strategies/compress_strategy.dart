import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:image_picker/image_picker.dart';

const _maxBytes = 1024 * 1024; // 1 MB cap

abstract class CompressStrategy {
  const CompressStrategy();

  Future<File> run(XFile source);

  @protected
  Future<File> ensureUnderLimit(
    XFile picked, {
    required List<int> qualities,
  }) async {
    final original = File(picked.path);
    if (!await original.exists()) return original;

    final length = await original.length();
    if (length <= _maxBytes) return original;

    File best = original;
    for (final quality in qualities) {
      final targetPath = _deriveCompressedPath(original.path, quality);
      final targetFile = File(targetPath);
      if (await targetFile.exists()) await targetFile.delete();

      final compressed = await FlutterImageCompress.compressAndGetFile(
        original.path,
        targetPath,
        quality: quality,
        minWidth: 1600,
        minHeight: 1600,
      );
      if (compressed == null) continue;

      final candidate = File(compressed.path);
      final candidateSize = await candidate.length();
      best = candidate;
      if (candidateSize <= _maxBytes) break;
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
}

class FastCompressStrategy extends CompressStrategy {
  const FastCompressStrategy();

  @override
  Future<File> run(XFile source) {
    return ensureUnderLimit(
      source,
      qualities: const [75, 62, 50, 38],
    );
  }
}

class QualityCompressStrategy extends CompressStrategy {
  const QualityCompressStrategy();

  @override
  Future<File> run(XFile source) {
    return ensureUnderLimit(
      source,
      qualities: const [90, 82, 74, 66, 58, 50, 40, 30, 22],
    );
  }
}

class OriginalCompressStrategy extends CompressStrategy {
  const OriginalCompressStrategy();

  @override
  Future<File> run(XFile source) async => File(source.path);
}

class CompressionContext {
  final CompressStrategy fastStrategy;
  final CompressStrategy qualityStrategy;
  final CompressStrategy originalStrategy;
  final bool forceOriginal;

  const CompressionContext({
    required this.fastStrategy,
    required this.qualityStrategy,
    required this.originalStrategy,
    this.forceOriginal = false,
  });

  CompressStrategy choose({required ImageSource source}) {
    if (forceOriginal) {
      return originalStrategy;
    }
    if (kIsWeb) {
      return fastStrategy;
    }
    if (source == ImageSource.camera) {
      return qualityStrategy;
    }
    return fastStrategy;
  }
}
