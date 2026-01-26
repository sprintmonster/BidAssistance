import { useState } from "react";
import { X } from "lucide-react";
import type { PostCategory } from "../types/community";

export type NewPostDraftForm = {
    title: string;
    content: string;
    category: PostCategory;
    files: File[];
};

interface NewPostFormProps {
	onSubmit: (draft: NewPostDraftForm) => void;
	onCancel: () => void;
}

export function NewPostForm({ onSubmit, onCancel }: NewPostFormProps) {
	const [title, setTitle] = useState("");
	const [content, setContent] = useState("");
	const [category, setCategory] = useState<PostCategory>("question");
	const [files, setFiles] = useState<File[]>([]);

	const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const selected = Array.from(e.target.files ?? []);
		if (selected.length === 0) return;

		setFiles((prev) => {
			const next = [...prev];
			for (const f of selected) {
				if (!next.some((x) => x.name === f.name && x.size === f.size)) next.push(f);
			}
			return next;
		});

		e.target.value = "";
	};

	const removeFile = (idx: number) => {
		setFiles((prev) => prev.filter((_, i) => i !== idx));
	};

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (!title.trim() || !content.trim()) return;

		onSubmit({
			title: title.trim(),
			content: content.trim(),
			category,
			files,
		});
	};

	return (
		<div className="bg-white rounded-lg border border-gray-200 p-8">
			<h2 className="text-2xl font-bold text-gray-900 mb-6">새 글 작성</h2>

			<form onSubmit={handleSubmit} className="space-y-6">
				<div>
					<label className="block text-sm font-medium text-gray-700 mb-2">카테고리</label>
					<div className="flex gap-3">
						<button type="button" onClick={() => setCategory("question")}
							className={`px-4 py-2 rounded-lg font-medium transition-colors ${
								category === "question" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
							}`}>
							질문
						</button>
						<button type="button" onClick={() => setCategory("info")}
							className={`px-4 py-2 rounded-lg font-medium transition-colors ${
								category === "info" ? "bg-green-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
							}`}>
							정보
						</button>
						<button type="button" onClick={() => setCategory("review")}
							className={`px-4 py-2 rounded-lg font-medium transition-colors ${
								category === "review" ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
							}`}>
							후기
						</button>
						<button type="button" onClick={() => setCategory("discussion")}
							className={`px-4 py-2 rounded-lg font-medium transition-colors ${
								category === "discussion" ? "bg-orange-600 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"
							}`}>
							토론
						</button>
					</div>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 mb-2">제목</label>
					<input
						type="text"
						value={title}
						onChange={(e) => setTitle(e.target.value)}
						placeholder="제목을 입력하세요"
						className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
						required
					/>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 mb-2">내용</label>
					<textarea
						value={content}
						onChange={(e) => setContent(e.target.value)}
						placeholder="내용을 입력하세요"
						className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
						rows={12}
						required
					/>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 mb-2">첨부파일</label>
					<input
						type="file"
						multiple
						accept="image/*,application/pdf"
						onChange={handleFilesChange}
						className="block w-full text-sm text-gray-600
							file:mr-4 file:py-2 file:px-4
							file:rounded-lg file:border-0
							file:text-sm file:font-semibold
							file:bg-gray-100 file:text-gray-700
							hover:file:bg-gray-200"
					/>

					{files.length > 0 && (
						<div className="mt-3 space-y-2">
							<div className="text-sm text-gray-600">선택된 파일 {files.length}개</div>

							<div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
								{files.map((f, idx) => {
									const isImage = f.type.startsWith("image/");
									const previewUrl = isImage ? URL.createObjectURL(f) : null;

									return (
										<div key={`${f.name}-${f.size}-${idx}`} className="border rounded-lg p-2 bg-gray-50">
											{isImage && previewUrl ? (
												<img
													src={previewUrl}
													alt={f.name}
													className="w-full h-24 object-cover rounded-md"
													onLoad={() => URL.revokeObjectURL(previewUrl)}
												/>
											) : (
												<div className="w-full h-24 flex items-center justify-center text-xs text-gray-500 bg-white rounded-md">
													{f.type || "file"}
												</div>
											)}

											<div className="mt-2 flex items-start justify-between gap-2">
												<div className="min-w-0">
													<div className="text-xs font-medium text-gray-800 truncate">{f.name}</div>
													<div className="text-[11px] text-gray-500">{Math.round(f.size / 1024)} KB</div>
												</div>

												<button
													type="button"
													onClick={() => removeFile(idx)}
													className="text-gray-400 hover:text-gray-700"
													aria-label="remove"
												>
													<X className="w-4 h-4" />
												</button>
											</div>
										</div>
									);
								})}
							</div>
						</div>
					)}
				</div>

				<div className="flex items-center justify-end gap-2">
					<button
						type="button"
						onClick={onCancel}
						className="px-4 py-2 rounded-lg border hover:bg-gray-50"
					>
						취소
					</button>
					<button
						type="submit"
						className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700"
					>
						등록
					</button>
				</div>
			</form>
		</div>
	);
}
