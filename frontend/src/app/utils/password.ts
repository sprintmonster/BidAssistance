export type PasswordRule = {
	key: string;
	label: string;
	ok: boolean;
};

export type PasswordPolicyContext = {
	user_id?: string;
	nickname?: string;
};

function has_alpha(s: string) {
	return /[A-Za-z]/.test(s);
}

function has_digit(s: string) {
	return /[0-9]/.test(s);
}

function has_special(s: string) {
	// 공백 제외 특수문자
	return /[^A-Za-z0-9\s]/.test(s);
}

function has_repeated_char(s: string, n: number) {
	const re = new RegExp(`(.)\\1{${n - 1},}`);
	return re.test(s);
}

function normalize(s: string) {
	return (s || "").toLowerCase();
}

export function get_password_rules(password: string, ctx?: PasswordPolicyContext): PasswordRule[] {
	const p = password || "";
	const a = has_alpha(p);
	const d = has_digit(p);
	const sp = has_special(p);
	const kinds = [a, d, sp].filter(Boolean).length;

	// 가이드라인: (영문/숫자/특수 중) 2종 조합이면 10자리 이상, 3종이면 8자리 이상
	const length_ok = (kinds >= 3 && p.length >= 8) || (kinds >= 2 && p.length >= 10);
	const combo_ok = kinds >= 2;
	const repeat_ok = !has_repeated_char(p, 4);

	const user_id = normalize(ctx?.user_id || "");
	const nickname = normalize(ctx?.nickname || "");
	const p_norm = normalize(p);
	const not_contains_id = user_id ? !p_norm.includes(user_id) : true;
	const not_contains_nick = nickname ? !p_norm.includes(nickname) : true;

	return [
		{
			key: "length",
			label: "길이: 2종 조합은 10자 이상, 3종 조합은 8자 이상",
			ok: length_ok,
		},
		{
			key: "combo",
			label: "영문/숫자/특수문자 중 2종 이상 포함",
			ok: combo_ok,
		},
		{
			key: "repeat",
			label: "동일 문자 4회 이상 연속 사용 금지",
			ok: repeat_ok,
		},
		{
			key: "no_id",
			label: "이메일(아이디)와 유사한 비밀번호 사용 금지",
			ok: not_contains_id,
		},
		{
			key: "no_nick",
			label: "닉네임과 유사한 비밀번호 사용 금지",
			ok: not_contains_nick,
		},
	];
}

export function is_password_valid(password: string, ctx?: PasswordPolicyContext): boolean {
	return get_password_rules(password, ctx).every((r) => r.ok);
}
